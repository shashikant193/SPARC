/**
 * @file    electronDensity.c
 * @brief   This file contains the functions for calculating electron density.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * @Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#include <complex.h> 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <math.h>

#include "electronicGroundState.h"
#include "electronDensity.h"
#include "eigenSolver.h"
#include "eigenSolverKpt.h" 
#include "isddft.h"
#include "sq3.h"
#include "cs.h"
#include "ddbp.h"
#include "extFPMD.h"
#include "spdft.h"

/*
@ brief: Main function responsible to find electron density
*/
void Calculate_elecDens(int rank, SPARC_OBJ *pSPARC, int SCFcount, double error){
    int i, DMnd;
    DMnd = pSPARC->Nd_d_dmcomm;
    double *rho = (double *) calloc(DMnd * (2*pSPARC->Nspinor-1), sizeof(double)); 
    double *mag = (double *) calloc(DMnd * pSPARC->Nmag, sizeof(double));   

#ifdef DEBUG
    double t1 = MPI_Wtime();
#endif
    
    // Currently only involves Chebyshev filtering eigensolver
    if (pSPARC->isGammaPoint) {
        eigSolve_CheFSI(rank, pSPARC, SCFcount, error);
        if (pSPARC->SQ3Flag == 1) {
            SubDensMat(pSPARC, pSPARC->Ds_cmc, pSPARC->Efermi, pSPARC->ChebComp);
        }
        if (pSPARC->DDBP_Flag == 1) {
            DDBP_INFO *DDBP_info = pSPARC->DDBP_info;
            int Nstates = pSPARC->Nstates;
            int nspin = pSPARC->Nspin_spincomm;
            int nkpt = pSPARC->Nkpts_kptcomm;
            Calculate_density_psi_DDBP(
                DDBP_info->n_elem_elemcomm, DDBP_info->elem_list,
                DDBP_info->psi, DDBP_info->rho, pSPARC->occ, pSPARC->dV,
                pSPARC->isGammaPoint, pSPARC->spin_typ, nspin, nkpt, Nstates,
                pSPARC->spin_start_indx, DDBP_info->band_start_index,
                DDBP_info->band_end_index, DDBP_info->elemcomm
            );

        #ifdef CHECK_RHO
            // check if \int{rho} = Nelectron
            double int_rho = 0.0;
            for (int k = 0; k < DDBP_info->n_elem_elemcomm; k++) {
                DDBP_ELEM *E_k = &DDBP_info->elem_list[k];
                int nd_k = E_k->nd_d;
                double *rho_k = DDBP_info->rho[k];
                for (int i = 0; i < nd_k; i++) {
                    int_rho += rho_k[i];
                }
            }
            int_rho *= pSPARC->dV;
            MPI_Allreduce(MPI_IN_PLACE, &int_rho, 1, MPI_DOUBLE, MPI_SUM, DDBP_info->bandcomm);
            double sum_occ = 0.0;
            for (int i = 0; i < Nstates; i++) {
                sum_occ += 2.0*pSPARC->occ[i];
            }
            printf("rank = %2d, checking rho: sum_occ = %.16f, int_rho = %.16f\n", rank, sum_occ, int_rho);
            // warning: this is only for checking spin-unpolarized test
            // sleep(1);
            if (pSPARC->spin_typ == 0 && rank == 0)
                assert(fabs(int_rho - sum_occ) < 1e-10);
        #endif
        }
    } else {
        eigSolve_CheFSI_kpt(rank, pSPARC, SCFcount, error);
    }

    CalculateDensity_psi(pSPARC, rho + (pSPARC->Nspinor-1)*DMnd);

	// Add homogenous electron density in spDFT method
	if (pSPARC->spDFT_Flag == 1) {
		spDFT_electrondensity_homo(pSPARC, rho + (pSPARC->Nspinor-1)*DMnd);
	}

    if (pSPARC->Nspinor > 1) {
        // calculate total electron density
        for (i = 0; i < DMnd; i++) {
            rho[i] = rho[DMnd+i] + rho[2*DMnd+i]; 
        }
    }

    // add high energy electron density for ext-FPMD method
    if (pSPARC->ext_FPMD_Flag != 0) {
        highE_rho_extFPMD(pSPARC, 1.0, rho, pSPARC->Nd_d_dmcomm);
    }

    if (pSPARC->spin_typ == 1) {
        Calculate_Magz(pSPARC, DMnd, mag, rho+DMnd, rho+2*DMnd); // magz
    }

    if (pSPARC->spin_typ == 2) {
        // magx, magy
        Calculate_Magx_Magy_psi(pSPARC, mag+DMnd); 
        // magz
        Calculate_Magz(pSPARC, DMnd, mag+3*DMnd, rho+DMnd, rho+2*DMnd); 
        // magnorm
        Calculate_Magnorm(pSPARC, DMnd, mag+DMnd, mag+2*DMnd, mag+3*DMnd, mag); 
        // update rhod11 rhod22
        Calculate_diagonal_Density(pSPARC, DMnd, mag, rho, rho+DMnd, rho+2*DMnd); 
    }

#ifdef DEBUG
    double t2 = MPI_Wtime();
    if(!rank) printf("rank = %d, Calculating density and magnetization took %.3f ms\n",rank,(t2-t1)*1e3);       
    if(!rank) printf("rank = %d, starting to transfer density and magnetization ...\n",rank);
#endif

    // transfer density from psi-domain to phi-domain
#ifdef DEBUG
    t1 = MPI_Wtime();
#endif
    
    if (pSPARC->DDBP_Flag == 1) {    
        for (int i = 0; i < pSPARC->Nspdentd; i++) {
            // transfter density from elem distribution to domain distribution
            DDBP_INFO *DDBP_info = pSPARC->DDBP_info;
            // element distribution to domain distribution
            int gridsizes[3] = {pSPARC->Nx, pSPARC->Ny, pSPARC->Nz};
            int BCs[3] = {pSPARC->BCx, pSPARC->BCy, pSPARC->BCz};
            int dmcomm_phi_dims[3] = {pSPARC->npNdx_phi, pSPARC->npNdy_phi, pSPARC->npNdz_phi};
            int send_ncol = DDBP_info->bandcomm_index == 0 ? 1 : 0;
            int recv_ncol = 1;
            int Edims[3] = {DDBP_info->Nex, DDBP_info->Ney, DDBP_info->Nez};
            E2D_INFO E2D_info;
            E2D_Init(&E2D_info, Edims, DDBP_info->n_elem_elemcomm, DDBP_info->elem_list,
                gridsizes, BCs, 1,
                0, send_ncol, DDBP_info->elemcomm, DDBP_info->npband, DDBP_info->elemcomm_index,
                DDBP_info->bandcomm, DDBP_info->npelem, DDBP_info->bandcomm_index,
                0, recv_ncol, pSPARC->DMVertices, MPI_COMM_SELF, 1, pSPARC->dmcomm_phi,
                &dmcomm_phi_dims[0], 0, pSPARC->kptcomm
            );

            E2D_Iexec(&E2D_info, (const void **) DDBP_info->rho);
            E2D_Wait(&E2D_info, pSPARC->electronDens + i*pSPARC->Nd_d);
            E2D_Finalize(&E2D_info);
        }
    } else {
        for (i = 0; i < pSPARC->Nspdentd; i++)
            TransferDensity(pSPARC, rho + i*DMnd, pSPARC->electronDens + i*pSPARC->Nd_d);
        
        for (i = 0; i < pSPARC->Nmag; i++)
            TransferDensity(pSPARC, mag + i*DMnd, pSPARC->mag + i*pSPARC->Nd_d);
    }

#ifdef DEBUG
    t2 = MPI_Wtime();
    if(!rank) printf("rank = %d, Transfering density and magnetization took %.3f ms\n", rank, (t2 - t1) * 1e3);
#endif

#ifdef CHECK_RHO
    // check electron density
    // check if \int{rho} = Nelectron
    double int_rho = 0.0;
    for (int i = 0; i < pSPARC->Nd_d; i++) {
        if (pSPARC->CyclixFlag) 
            int_rho += pSPARC->electronDens[i] * pSPARC->Intgwt_phi[i];
        else
            int_rho += pSPARC->electronDens[i] * pSPARC->dV;
    }
    
    if (pSPARC->dmcomm_phi != MPI_COMM_NULL)
        MPI_Allreduce(MPI_IN_PLACE, &int_rho, 1, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm_phi);
    double sum_occ = 0.0;
    for (int i = 0; i < pSPARC->Nstates; i++) {
        sum_occ += pSPARC->occfac * pSPARC->occ[i];
    }
    if (pSPARC->ext_FPMD_Flag != 0) {
        usleep(20000);
	    double HighECharge = calculate_highE_Charge_extFPMD(pSPARC, pSPARC->Efermi);
        printf("== CHECK_RHO ==: rank = %d, Efermi = %f, HighECharge = %f\n", rank, pSPARC->Efermi, HighECharge);
        sum_occ += HighECharge;
    }
    printf("rank = %2d, after transfering rho: sum_occ = %.16f, int_rho = %.16f\n", rank, sum_occ, int_rho);
    // warning: this is only for checking spin-unpolarized test
    if (pSPARC->spin_typ == 0 && rank == 0)
        assert(fabs(int_rho - sum_occ) < 1e-8 * sum_occ);
#endif

    free(rho);
    free(mag);
}


/**
 * @brief   Calculate electron density with given states in psi-domain.
 *
 *          Note that here rho is distributed in psi-domain, which needs
 *          to be transmitted to phi-domain for solving the poisson 
 *          equation.
 */
void CalculateDensity_psi(SPARC_OBJ *pSPARC, double *rho)
{
    if (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;

    int i, n, k, Ns, count, nstart, nend, spinor, DMnd;
    double g_nk;
    Ns = pSPARC->Nstates;
    nstart = pSPARC->band_start_indx;
    nend = pSPARC->band_end_indx;
    DMnd = pSPARC->Nd_d_dmcomm;
    int Nspinor = pSPARC->Nspinor;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#ifdef DEBUG
    double t1, t2;
    t1 = MPI_Wtime();
#endif

    if (pSPARC->SQ3Flag == 1) {
        update_rho_sq3(pSPARC, rho);
    } else if (pSPARC->CS_Flag == 1) {
        update_rho_cs(pSPARC, rho);
    } else {
        // calculate rho based on local bands
        count = 0;
        for (k = 0; k < pSPARC->Nkpts_kptcomm; k++) {
            for (n = nstart; n <= nend; n++) {
                double woccfac = pSPARC->occfac * (pSPARC->kptWts_loc[k] / pSPARC->Nkpts);
                for (spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor ++) {
                    int spinor_g = spinor + pSPARC->spinor_start_indx;
                    double *occ = pSPARC->occ + k*Ns; 
                    if (pSPARC->spin_typ == 1) occ += spinor*Ns*pSPARC->Nkpts_kptcomm;
                    g_nk = woccfac * occ[n];

                    if (pSPARC->isGammaPoint) {
                        for (i = 0; i < DMnd; i++) {
                            rho[i+spinor_g*DMnd] += g_nk * pSPARC->Xorb[count] * pSPARC->Xorb[count];
                            count++;
                        }
                    } else {
                        for (i = 0; i < DMnd; i++) {
                            rho[i+spinor_g*DMnd] += g_nk * (creal(pSPARC->Xorb_kpt[count]) * creal(pSPARC->Xorb_kpt[count])
                                                        + cimag(pSPARC->Xorb_kpt[count]) * cimag(pSPARC->Xorb_kpt[count]));
                            count++;
                        }
                    }
                }
            }
        }
    }

#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("rank = %d, --- Calculate rho: sum over local bands took %.3f ms\n", rank, (t2-t1)*1e3);
    t1 = MPI_Wtime();
#endif
    
    // sum over spin comm group
    if(pSPARC->npspin > 1) {        
        MPI_Allreduce(MPI_IN_PLACE, rho, Nspinor*DMnd, MPI_DOUBLE, MPI_SUM, pSPARC->spin_bridge_comm);        
    }

#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("rank = %d, --- Calculate rho: reduce over all spin_comm took %.3f ms\n", rank, (t2-t1)*1e3);
    t1 = MPI_Wtime();
#endif

    // sum over all k-point groups
    if (pSPARC->npkpt > 1) {            
        MPI_Allreduce(MPI_IN_PLACE, rho, Nspinor*DMnd, MPI_DOUBLE, MPI_SUM, pSPARC->kpt_bridge_comm);
    }

#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("rank = %d, --- Calculate rho: reduce over all kpoint groups took %.3f ms\n", rank, (t2-t1)*1e3);
    t1 = MPI_Wtime();
#endif
    
    // sum over all band groups 
    if (pSPARC->npband) {
        MPI_Allreduce(MPI_IN_PLACE, rho, Nspinor*DMnd, MPI_DOUBLE, MPI_SUM, pSPARC->blacscomm);
    }

#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("rank = %d, --- Calculate rho: reduce over all band groups took %.3f ms\n", rank, (t2-t1)*1e3);
    t1 = MPI_Wtime();
#endif

    if (!pSPARC->CyclixFlag) {
        double vscal = 1.0 / pSPARC->dV;
        // scale electron density by 1/dV        
        for (i = 0; i < Nspinor*DMnd; i++) {
            rho[i] *= vscal; 
        }
        
    #ifdef DEBUG
        t2 = MPI_Wtime();
        if (!rank) printf("rank = %d, --- Scale rho: scale by 1/dV took %.3f ms\n", rank, (t2-t1)*1e3);
    #endif
    }
}



/**
 * @brief   Calculate off-diagonal electron density with given states in psi-domain.
 *
 */
void Calculate_Magx_Magy_psi(SPARC_OBJ *pSPARC, double *mag)
{
    if (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;

    int i, n, k, Ns, count, nstart, nend, DMnd;    
    Ns = pSPARC->Nstates;
    nstart = pSPARC->band_start_indx;
    nend = pSPARC->band_end_indx;
    DMnd = pSPARC->Nd_d_dmcomm;    

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#ifdef DEBUG
    double t1, t2;
    t1 = MPI_Wtime();
#endif    

    // calculate rho based on local bands
    count = 0;
    for (k = 0; k < pSPARC->Nkpts_kptcomm; k++) {
        for (n = nstart; n <= nend; n++) {
            double woccfac = pSPARC->occfac * (pSPARC->kptWts_loc[k] / pSPARC->Nkpts);
            double g_nk = woccfac * pSPARC->occ[n + k*Ns];
            for (i = 0; i < DMnd; i++) {
                double _Complex rho_odd = pSPARC->Xorb_kpt[count] * conj(pSPARC->Xorb_kpt[count+DMnd]);
                mag[i] += 2 * g_nk * creal(rho_odd);      // magx
                mag[i+DMnd] -= 2 * g_nk * cimag(rho_odd); // magy
                count ++;
            }
            count += DMnd;
        }
    }

#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("rank = %d, --- Calculate magx, magy: sum over local bands took %.3f ms\n", rank, (t2-t1)*1e3);
    t1 = MPI_Wtime();
#endif
    
    // sum over spin comm group
    if(pSPARC->npspin > 1) {        
        MPI_Allreduce(MPI_IN_PLACE, mag, 2*DMnd, MPI_DOUBLE, MPI_SUM, pSPARC->spin_bridge_comm);        
    }

#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("rank = %d, --- Calculate magx, magy: reduce over all spin_comm took %.3f ms\n", rank, (t2-t1)*1e3);
    t1 = MPI_Wtime();
#endif

    // sum over all k-point groups
    if (pSPARC->npkpt > 1) {            
        MPI_Allreduce(MPI_IN_PLACE, mag, 2*DMnd, MPI_DOUBLE, MPI_SUM, pSPARC->kpt_bridge_comm);
    }

#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("rank = %d, --- Calculate magx, magy: reduce over all kpoint groups took %.3f ms\n", rank, (t2-t1)*1e3);
    t1 = MPI_Wtime();
#endif
    
    // sum over all band groups 
    if (pSPARC->npband) {
        MPI_Allreduce(MPI_IN_PLACE, mag, 2*DMnd, MPI_DOUBLE, MPI_SUM, pSPARC->blacscomm);
    }

#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("rank = %d, --- Calculate magx, magy: reduce over all band groups took %.3f ms\n", rank, (t2-t1)*1e3);
    t1 = MPI_Wtime();
#endif

    if (!pSPARC->CyclixFlag) {
        double vscal = 1.0 / pSPARC->dV;
        // scale mag by 1/dV
        for (i = 0; i < 2*DMnd; i++) {
            mag[i] *= vscal; 
        }
        
    #ifdef DEBUG
        t2 = MPI_Wtime();
        if (!rank) printf("rank = %d, --- Scale mag: scale by 1/dV took %.3f ms\n", rank, (t2-t1)*1e3);
    #endif
    }
}

/*
@ brief: calculate magz
*/ 
void Calculate_Magz(SPARC_OBJ *pSPARC, int DMnd, double *magz, double *rhoup, double *rhodw)
{
    for (int i = 0; i < DMnd; i++) {
        magz[i] = rhoup[i] - rhodw[i];
    }
}

/*
@ brief: calculate norm of magnetization
*/ 
void Calculate_Magnorm(SPARC_OBJ *pSPARC, int DMnd, double *magx, double *magy, double *magz, double *magnorm)
{
    for (int i = 0; i < DMnd; i++) {
        magnorm[i] = sqrt(magx[i]*magx[i] + magy[i]*magy[i] + magz[i]*magz[i]);
    }
}


void Calculate_diagonal_Density(SPARC_OBJ *pSPARC, int DMnd, double *magnorm, double *rho_tot, double *rho11, double *rho22)
{
    for (int i = 0; i < DMnd; i++) {
        rho11[i] = 0.5*(rho_tot[i] + magnorm[i]);
        rho22[i] = 0.5*(rho_tot[i] - magnorm[i]);
    }
}