/***
 * @file    exactExchange.c
 * @brief   This file contains the functions for Exact Exchange.
 *
 * @authors Xin Jing <xjing30@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <mpi.h>
#include <complex.h>
#include <limits.h>
/** BLAS and LAPACK routines */
#ifdef USE_MKL
    #define MKL_Complex16 double _Complex
    #include <mkl.h>
#else
    #include <cblas.h>
    #include <lapacke.h>
#endif
/** ScaLAPACK routines */
#ifdef USE_MKL
    #include "blacs.h"     // Cblacs_*
    #include <mkl_blacs.h>
    #include <mkl_pblas.h>
    #include <mkl_scalapack.h>
#endif
#ifdef USE_SCALAPACK
    #include "blacs.h"     // Cblacs_*
    #include "scalapack.h" // ScaLAPACK functions
#endif
#ifdef USE_FFTW
    #include <fftw3.h>
#endif

#include "exactExchange.h"
#include "lapVecRoutines.h"
#include "linearSolver.h"
#include "exactExchangeKpt.h"
#include "tools.h"
#include "parallelization.h"
#include "electronicGroundState.h"
#include "exchangeCorrelation.h"
#include "exactExchangeInitialization.h"
#include "electrostatics.h"
#include "sqDensity.h"
#include "sqParallelization.h"
#include "sqExactExchange.h"
#include "kroneckerLaplacian.h"

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))


#define TEMP_TOL (1e-12)


/**
 * @brief   Outer loop of SCF using Vexx (exact exchange potential)
 */
void Exact_Exchange_loop(SPARC_OBJ *pSPARC) {
    int i, rank;
    double t1, t2, ACE_time = 0.0;
    FILE *output_fp;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int DMnd = pSPARC->Nd_d;

    /************************ Exact exchange potential parameters ************************/
    int count_xx = 0;
    double Eexx_pre = pSPARC->Eexx, err_Exx = pSPARC->TOL_FOCK + 1;
    pSPARC->Exxtime = pSPARC->ACEtime = 0.0;

    /************************* Update Veff copied from SCF code **************************/
    #ifdef DEBUG
    if(!rank) 
        printf("\nStart evaluating Exact Exchange !\n");
    #endif  
    
    // for the first outer loop with SQ.
    if (pSPARC->SQFlag) {
        t1 = MPI_Wtime();
        pSPARC->usefock--;
        calculate_density_matrix_SQ(pSPARC);
        pSPARC->usefock++;
        t2 = MPI_Wtime();
        #ifdef DEBUG
        if(!rank) 
            printf("rank = %d, calculating density matrix took %.3f ms\n",rank,(t2-t1)*1e3); 
        #endif 
    }

    // calculate xc potential (LDA), "Vxc"
    t1 = MPI_Wtime(); 
    Calculate_Vxc(pSPARC);
    t2 = MPI_Wtime();
    #ifdef DEBUG
    if (rank == 0) printf("rank = %d, XC calculation took %.3f ms\n", rank, (t2-t1)*1e3); 
    #endif 
    
    t1 = MPI_Wtime();
    // calculate Veff_loc_dmcomm_phi = phi + Vxc in "phi-domain"
    Calculate_Veff_loc_dmcomm_phi(pSPARC);

    // initialize mixing_hist_xk (and mixing_hist_xkm1)
    Update_mixing_hist_xk(pSPARC);

    if (pSPARC->SQFlag == 1) {
        TransferVeff_phi2sq(pSPARC, pSPARC->Veff_loc_dmcomm_phi, pSPARC->pSQ->Veff_loc_SQ);
    } else {
        // transfer Veff_loc from "phi-domain" to "psi-domain"
        for (i = 0; i < pSPARC->Nspden; i++)
            Transfer_Veff_loc(pSPARC, pSPARC->Veff_loc_dmcomm_phi + i*DMnd, pSPARC->Veff_loc_dmcomm + i*pSPARC->Nd_d_dmcomm);
    }

    t2 = MPI_Wtime();
    #ifdef DEBUG
    if(!rank) 
        printf("rank = %d, Veff calculation and Bcast (non-blocking) took %.3f ms\n",rank,(t2-t1)*1e3); 
    #endif 

    /******************************* Hartre-Fock outer loop ******************************/
    while (count_xx < pSPARC->MAXIT_FOCK) {
    #ifdef DEBUG
    if(!rank) 
        printf("\nHartree-Fock Outer Loop: %d \n",count_xx + 1);
    #endif  

        if (pSPARC->SQFlag) {
            t1 = MPI_Wtime();
            collect_col_of_Density_matrix(pSPARC);
            if (pSPARC->ExxAcc == 1 && pSPARC->SQ_gauss_hybrid_mem == 1) {
                compute_exx_potential_SQ(pSPARC);
            }
            t2 = MPI_Wtime();
            #ifdef DEBUG
            if (pSPARC->ExxAcc == 1 && pSPARC->SQ_gauss_hybrid_mem == 1) {
                if(!rank) printf("\nCollecting columns of density matrix and calculating exx potential for SQ hybrid took : %.3f ms\n", (t2-t1)*1e3);
            } else {
                if(!rank) printf("\nCollecting columns of density matrix for SQ hybrid took : %.3f ms\n", (t2-t1)*1e3);
            }
            #endif 
            pSPARC->ACEtime += (t2 - t1);
            ACE_time = (t2 - t1);
        } else {
            if (pSPARC->ExxAcc == 0) {
                if (pSPARC->isGammaPoint == 1) {
                    // Gathering all outer orbitals into each band comm
                    t1 = MPI_Wtime();
                    gather_psi_occ_outer(pSPARC, pSPARC->psi_outer, pSPARC->occ_outer);
                    t2 = MPI_Wtime();
                    #ifdef DEBUG
                    if(!rank) 
                        printf("\nGathering all bands of psi_outer to each dmcomm took : %.3f ms\n", (t2-t1)*1e3);
                    #endif 
                } else {
                    // Gathering all outer orbitals and outer occ
                    t1 = MPI_Wtime();
                    gather_psi_occ_outer_kpt(pSPARC, pSPARC->psi_outer_kpt, pSPARC->occ_outer);
                    t2 = MPI_Wtime();
                    #ifdef DEBUG
                    if(!rank) 
                        printf("\nGathering all bands and all kpoints of psi_outer and occupations to each dmcomm took : %.3f ms\n", (t2-t1)*1e3);
                    #endif 
                }
            } else if (pSPARC->ExxAcc == 1) {
                #ifdef DEBUG
                if(!rank) printf("\nStart to create ACE operator!\n");
                #endif  
                t1 = MPI_Wtime();
                // create ACE operator 
                if (pSPARC->isGammaPoint == 1) {
                    allocate_ACE(pSPARC);                
                    ACE_operator(pSPARC, pSPARC->Xorb, pSPARC->occ, pSPARC->Xi);
                } else {
                    gather_psi_occ_outer_kpt(pSPARC, pSPARC->psi_outer_kpt, pSPARC->occ_outer);
                    allocate_ACE_kpt(pSPARC);                
                    ACE_operator_kpt(pSPARC, pSPARC->Xorb_kpt, pSPARC->occ_outer, pSPARC->Xi_kpt);
                }
                t2 = MPI_Wtime();
                pSPARC->ACEtime += (t2 - t1);
                ACE_time = (t2 - t1);
                #ifdef DEBUG
                if(!rank) printf("\nCreating ACE operator took %.3f ms!\n", (t2 - t1)*1e3);
                #endif
            }
        }

        // transfer psi_outer from "psi-domain" to "phi-domain" in No-ACE case 
        // transfer Xi from "psi-domain" to "phi-domain" in ACE case 
        t1 = MPI_Wtime();
        if (!pSPARC->SQFlag && pSPARC->ExxAcc == 0) {
            if (pSPARC->isGammaPoint == 1) {
                Transfer_dmcomm_to_kptcomm_topo(pSPARC, pSPARC->Nspinor_spincomm, pSPARC->Nstates, pSPARC->psi_outer, pSPARC->psi_outer_kptcomm_topo, sizeof(double));    
            } else {
                Transfer_dmcomm_to_kptcomm_topo(pSPARC, pSPARC->Nspinor_spincomm, pSPARC->Nstates*pSPARC->Nkpts_hf_red, pSPARC->psi_outer_kpt, pSPARC->psi_outer_kptcomm_topo_kpt, sizeof(double _Complex));
            }
            
            t2 = MPI_Wtime();
            #ifdef DEBUG
            if(!rank) 
                printf("\nTransfering all bands of psi_outer to kptcomm_topo took : %.3f ms\n", (t2-t1)*1e3);
            #endif  
        } else if (!pSPARC->SQFlag && pSPARC->ExxAcc == 1) {
            if (pSPARC->isGammaPoint == 1) {
                Transfer_dmcomm_to_kptcomm_topo(pSPARC, pSPARC->Nspinor_spincomm, pSPARC->Nstates_occ, pSPARC->Xi, pSPARC->Xi_kptcomm_topo, sizeof(double));
            } else {
                Transfer_dmcomm_to_kptcomm_topo(pSPARC, pSPARC->Nspinor_spincomm, pSPARC->Nstates_occ*pSPARC->Nkpts_kptcomm, pSPARC->Xi_kpt, pSPARC->Xi_kptcomm_topo_kpt, sizeof(double _Complex));                
            }

            t2 = MPI_Wtime();
            #ifdef DEBUG
            if(!rank) 
                printf("\nTransfering Xi to kptcomm_topo took : %.3f ms\n", (t2-t1)*1e3);
            #endif  
        }

        // compute exact exchange energy estimation with psi_outer
        // Eexx saves negative exact exchange energy without hybrid mixing
        if (pSPARC->SQFlag == 1) {
            exact_exchange_energy_SQ(pSPARC);
        } else {
            exact_exchange_energy(pSPARC);
        }

        if(!rank) {
            // write to .out file
            output_fp = fopen(pSPARC->OutFilename,"a");
            if (pSPARC->SQFlag == 1) {
                fprintf(output_fp,"\nNo.%d Exx outer loop. Basis timing: %.3f (sec)\n", count_xx + 1, ACE_time);
            } else if (pSPARC->ExxAcc == 0) {
                fprintf(output_fp,"\nNo.%d Exx outer loop. \n", count_xx + 1);
            } else {
                fprintf(output_fp,"\nNo.%d Exx outer loop. ACE timing: %.3f (sec)\n", count_xx + 1, ACE_time);
            }
            fclose(output_fp);
        }

        scf_loop(pSPARC);        

        Eexx_pre = pSPARC->Eexx;
        // update the final exact exchange energy
        pSPARC->Exc -= pSPARC->Eexx;
        pSPARC->Etot += 2 * pSPARC->Eexx;

        // compute exact exchange energy
        if (pSPARC->SQFlag == 1) {
            calculate_density_matrix_SQ(pSPARC);
            exact_exchange_energy_SQ(pSPARC);
        } else {
            exact_exchange_energy(pSPARC);
        }
        pSPARC->Exc += pSPARC->Eexx;
        pSPARC->Etot -= 2*pSPARC->Eexx;

        // error evaluation
        err_Exx = fabs(Eexx_pre - pSPARC->Eexx)/pSPARC->n_atom;
        MPI_Bcast(&err_Exx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);                 // TODO: Create bridge comm 
        if(!rank) {
            // write to .out file
            output_fp = fopen(pSPARC->OutFilename,"a");
            fprintf(output_fp,"Exx outer loop error: %.10e \n",err_Exx);
            fclose(output_fp);
        }
        if (err_Exx < pSPARC->TOL_FOCK && (count_xx+1) >= pSPARC->MINIT_FOCK) break;        
        
        count_xx ++;
    }

    #ifdef DEBUG
    if(!rank) 
        printf("\nFinished outer loop in %d steps!\n", count_xx);
    #endif  

    if (err_Exx > pSPARC->TOL_FOCK) {
        if(!rank) {
            printf("WARNING: EXX outer loop did not converge to desired accuracy!\n");
            // write to .out file
            output_fp = fopen(pSPARC->OutFilename,"a");
            fprintf(output_fp,"WARNING: EXX outer loop did not converge to desired accuracy!\n");
            fclose(output_fp);
        }
    }

    #ifdef DEBUG
    if(!rank && pSPARC->SQFlag && pSPARC->ExxAcc == 0) {
        printf("\n== Exact exchange Timing in SQ (Hsub routine) takes   %.3f ms\n", pSPARC->Exxtime*1e3);
    }
    if(!rank && pSPARC->SQFlag && pSPARC->ExxAcc == 1) {
        printf("\n== Exact exchange Timing in SQ (Hsub routine) takes %.3f ms\tcalculating potential takes %.3f ms\n", pSPARC->Exxtime*1e3, pSPARC->ACEtime*1e3);
    }
    if(!rank && !pSPARC->SQFlag && pSPARC->ExxAcc == 1) {
        printf("\n== Exact exchange Timing: creating ACE: %.3f ms\tapply ACE: %.3f ms\n",
            pSPARC->ACEtime*1e3, pSPARC->Exxtime*1e3);
    }
    if(!rank && !pSPARC->SQFlag && pSPARC->ExxAcc == 0) {
        printf("\n== Exact exchange Timing: apply Vx takes    %.3f ms\n", pSPARC->Exxtime*1e3);
    }
    #endif  
}


/**
 * @brief   Evaluating Exact Exchange potential
 *          
 *          This function basically prepares different variables for kptcomm_topo and dmcomm
 */
void exact_exchange_potential(SPARC_OBJ *pSPARC, double *X, int ldx, int ncol, int DMnd, double *Hx, int ldhx, int spin, MPI_Comm comm) 
{
    int rank, Lanczos_flag, dims[3];
    double *Xi, t1, t2, *occ;
    
    MPI_Comm_rank(comm, &rank);
    Lanczos_flag = (comm == pSPARC->kptcomm_topo) ? 1 : 0;
    /********************************************************************/

    int DMndsp = DMnd * pSPARC->Nspinor_spincomm;
    int occ_outer_shift = pSPARC->Nstates;

    t1 = MPI_Wtime();
    if (pSPARC->ExxAcc == 0) {
        if (Lanczos_flag == 0) {
            dims[0] = pSPARC->npNdx; dims[1] = pSPARC->npNdy; dims[2] = pSPARC->npNdz;
        } else {
            dims[0] = pSPARC->npNdx_kptcomm; dims[1] = pSPARC->npNdy_kptcomm; dims[2] = pSPARC->npNdz_kptcomm;
        }
        occ = (pSPARC->spin_typ == 1) ? (pSPARC->occ_outer + spin * occ_outer_shift) : pSPARC->occ_outer;
        double *psi_outer = (Lanczos_flag == 0) ? pSPARC->psi_outer + spin* DMnd : pSPARC->psi_outer_kptcomm_topo + spin* DMnd;
        evaluate_exact_exchange_potential(pSPARC, X, ldx, ncol, DMnd, dims, occ, psi_outer, DMndsp, Hx, ldhx, comm);
    } else {
        Xi = (Lanczos_flag == 0) ? pSPARC->Xi + spin * DMnd : pSPARC->Xi_kptcomm_topo + spin * DMnd;
        evaluate_exact_exchange_potential_ACE(pSPARC, X, ldx, ncol, DMnd, Xi, DMndsp, Hx, ldhx, comm);
    }

    t2 = MPI_Wtime();
    pSPARC->Exxtime +=(t2-t1);
}


/**
 * @brief   Evaluate Exact Exchange potential using non-ACE operator
 *          
 * @param X               The vectors premultiplied by the Fock operator
 * @param ncol            Number of columns of vector X
 * @param DMnd            Number of FD nodes in comm
 * @param dims            3 dimensions of comm processes grid
 * @param occ_outer       Full set of occ_outer occupations
 * @param psi_outer       Full set of psi_outer orbitals
 * @param Hx              Result of Hx plus fock operator times X 
 * @param comm            Communicator where the operation happens. dmcomm or kptcomm_topo
 */
void evaluate_exact_exchange_potential(SPARC_OBJ *pSPARC, double *X, int ldx, int ncol, int DMnd, int *dims, 
                                    double *occ_outer, double *psi_outer, int ldpo, double *Hx, int ldhx, MPI_Comm comm)
{
    int i, j, k, rank, Ns, num_rhs, *rhs_list_i, *rhs_list_j;
    int size, batch_num_rhs, NL, base, loop;
    double occ, *rhs, *Vi, exx_frac, occ_alpha;

    Ns = pSPARC->Nstates;
    exx_frac = pSPARC->exx_frac;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(comm, &size);
    /********************************************************************/
    rhs_list_i = (int*) calloc(ncol * Ns, sizeof(int)); 
    rhs_list_j = (int*) calloc(ncol * Ns, sizeof(int)); 
    assert(rhs_list_i != NULL && rhs_list_j != NULL);

    // Find the number of Poisson's equation required to be solved
    // Using the occupation threshold 1e-6
    int count = 0;
    for (i = 0; i < ncol; i++) {
        for (j = 0; j < Ns; j++) {
            if (occ_outer[j] > 1e-6) {
                rhs_list_i[count] = i;
                rhs_list_j[count] = j;
                count++;
            }
        }
    }
    num_rhs = count;

    if (num_rhs == 0) {
        free(rhs_list_i);
        free(rhs_list_j);
        return;
    }

    batch_num_rhs = pSPARC->ExxMemBatch == 0 ? 
                        num_rhs : pSPARC->ExxMemBatch * size;
    NL = (num_rhs - 1) / batch_num_rhs + 1;                                                // number of loops required                        
    rhs = (double *)malloc(sizeof(double) * DMnd * batch_num_rhs);                         // right hand sides of Poisson's equation
    Vi = (double *)malloc(sizeof(double) * DMnd * batch_num_rhs);                          // the solution for each rhs
    assert(rhs != NULL && Vi != NULL);

    /*************** Solve all Poisson's equation and apply to X ****************/    
    for (loop = 0; loop < NL; loop ++) {
        base = batch_num_rhs*loop;
        for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
            i = rhs_list_i[count];
            j = rhs_list_j[count];
            for (k = 0; k < DMnd; k++) {
                rhs[k + (count-base)*DMnd] = psi_outer[k + j*ldpo] * X[k + i*ldx];
            }
        }

        // Solve all Poisson's equation 
        poissonSolve(pSPARC, rhs, pSPARC->pois_const, count-base, DMnd, dims, Vi, comm);

        // Apply exact exchange potential to vector X
        for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
            i = rhs_list_i[count];
            j = rhs_list_j[count];
            occ = occ_outer[j];
            occ_alpha = occ * exx_frac;
            for (k = 0; k < DMnd; k++) {
                Hx[k + i*ldhx] -= occ_alpha * psi_outer[k + j*ldpo] * Vi[k + (count-base)*DMnd] / pSPARC->dV;
            }
        }
    }

    
    free(rhs);
    free(Vi);
    free(rhs_list_i);
    free(rhs_list_j);
}



/**
 * @brief   Evaluate Exact Exchange potential using ACE operator
 *          
 * @param X               The vectors premultiplied by the Fock operator
 * @param ncol            Number of columns of vector X
 * @param DMnd            Number of FD nodes in comm
 * @param Xi              Xi of ACE operator 
 * @param Hx              Result of Hx plus Vx times X
 * @param spin            Local spin index
 * @param comm            Communicator where the operation happens. dmcomm or kptcomm_topo
 */
void evaluate_exact_exchange_potential_ACE(SPARC_OBJ *pSPARC, double *X, int ldx, 
    int ncol, int DMnd, double *Xi, int ldxi, double *Hx, int ldhx, MPI_Comm comm) 
{
    int rank, size, Nstates_occ;
    Nstates_occ = pSPARC->Nstates_occ;
    double *Xi_times_psi = (double *) calloc(Nstates_occ * ncol, sizeof(double));
    assert(Xi_times_psi != NULL);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(comm, &size);
    /********************************************************************/

    // perform matrix multiplication Xi' * X using ScaLAPACK routines
    if (ncol != 1) {
        cblas_dgemm( CblasColMajor, CblasTrans, CblasNoTrans, Nstates_occ, ncol, DMnd,
                    1.0, Xi, ldxi, X, ldx, 0.0, Xi_times_psi, Nstates_occ);
    } else {
        cblas_dgemv( CblasColMajor, CblasTrans, DMnd, Nstates_occ, 1.0, 
                    Xi, ldxi, X, 1, 0.0, Xi_times_psi, 1);
    }

    if (size > 1) {
        // sum over all processors in dmcomm
        MPI_Allreduce(MPI_IN_PLACE, Xi_times_psi, Nstates_occ*ncol, 
                      MPI_DOUBLE, MPI_SUM, comm);
    }

    // perform matrix multiplication Xi * (Xi'*X) using ScaLAPACK routines
    if (ncol != 1) {
        cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, DMnd, ncol, Nstates_occ,
                    -pSPARC->exx_frac, Xi, ldxi, Xi_times_psi, Nstates_occ, 1.0, Hx, ldhx);
    } else {
        cblas_dgemv( CblasColMajor, CblasNoTrans, DMnd, Nstates_occ, -pSPARC->exx_frac, 
                    Xi, ldxi, Xi_times_psi, 1, 1.0, Hx, 1);
    }

    free(Xi_times_psi);
}


void exact_exchange_energy(SPARC_OBJ *pSPARC)
{
#ifdef DEBUG
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double t1, t2;
    t1 = MPI_Wtime();
#endif

    if (pSPARC->isGammaPoint == 1) {
        evaluate_exact_exchange_energy(pSPARC);
    } else {
        evaluate_exact_exchange_energy_kpt(pSPARC);
    }

#ifdef DEBUG
    t2 = MPI_Wtime();
if(!rank) 
    printf("\nEvaluating Exact exchange energy took: %.3f ms\nExact exchange energy %.6f.\n", (t2-t1)*1e3, pSPARC->Eexx);
#endif  
}

/**
 * @brief   Evaluate Exact Exchange Energy
 */
void evaluate_exact_exchange_energy(SPARC_OBJ *pSPARC) {
    if (pSPARC->spincomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    int i, j, k, grank, rank, size;
    int Ns, ncol, DMnd, DMndsp, dims[3], num_rhs, batch_num_rhs, NL, loop, base;
    double occ_i, occ_j, *rhs, *Vi, *psi_outer, temp, *occ_outer, *psi;
    MPI_Comm comm;

    DMnd = pSPARC->Nd_d_dmcomm;
    DMndsp = DMnd * pSPARC->Nspinor_spincomm;
    Ns = pSPARC->Nstates;
    ncol = pSPARC->Nband_bandcomm;
    comm = pSPARC->dmcomm;
    pSPARC->Eexx = 0.0;
    /********************************************************************/

    MPI_Comm_rank(MPI_COMM_WORLD, &grank);    
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (pSPARC->ExxAcc == 0) {
        dims[0] = pSPARC->npNdx; 
        dims[1] = pSPARC->npNdy; 
        dims[2] = pSPARC->npNdz;

        int *rhs_list_i, *rhs_list_j;
        rhs_list_i = (int*) calloc(ncol * Ns, sizeof(int)); 
        rhs_list_j = (int*) calloc(ncol * Ns, sizeof(int)); 
        assert(rhs_list_i != NULL && rhs_list_j != NULL);

        for (int spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor ++) {
            psi_outer = pSPARC->psi_outer + spinor * DMnd;
            occ_outer = (pSPARC->spin_typ == 1) ? (pSPARC->occ_outer + spinor * Ns) : pSPARC->occ_outer;
            psi = pSPARC->Xorb + spinor * DMnd;

            // Find the number of Poisson's equation required to be solved
            // Using the occupation threshold 1e-6
            int count = 0;
            for (i = 0; i < ncol; i++) {
                for (j = 0; j < Ns; j++) {
                    if (occ_outer[i] + occ_outer[j] > 1e-6) {
                        rhs_list_i[count] = i;
                        rhs_list_j[count] = j;
                        count++;
                    }
                }
            }
            num_rhs = count;
            if (num_rhs == 0) continue;            

            batch_num_rhs = pSPARC->ExxMemBatch == 0 ? 
                            num_rhs : pSPARC->ExxMemBatch * size;
        
            NL = (num_rhs - 1) / batch_num_rhs + 1;                                                // number of loops required
            rhs = (double *)malloc(sizeof(double) * DMnd * batch_num_rhs);                         // right hand sides of Poisson's equation
            Vi = (double *)malloc(sizeof(double) * DMnd * batch_num_rhs);                          // the solution for each rhs
            assert(rhs != NULL && Vi != NULL);

            for (loop = 0; loop < NL; loop ++) {
                base = batch_num_rhs*loop;
                for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
                    i = rhs_list_i[count];
                    j = rhs_list_j[count];
                    for (k = 0; k < DMnd; k++) {
                        rhs[k + (count-base)*DMnd] = psi_outer[k + j*DMndsp] * psi[k + i*DMndsp];
                    }
                }

                // Solve all Poisson's equation 
                poissonSolve(pSPARC, rhs, pSPARC->pois_const, count-base, DMnd, dims, Vi, comm);

                for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
                    i = rhs_list_i[count];
                    j = rhs_list_j[count];
                    
                    occ_i = occ_outer[i + pSPARC->band_start_indx];
                    occ_j = occ_outer[j];

                    // TODO: use a temp array to reduce the MPI_Allreduce time to 1
                    temp = 0.0;
                    for (k = 0; k < DMnd; k++){
                        temp += rhs[k + (count-base)*DMnd] * Vi[k + (count-base)*DMnd];
                    }
                    if (size > 1)
                        MPI_Allreduce(MPI_IN_PLACE, &temp, 1,  MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm);
                    pSPARC->Eexx += occ_i * occ_j * temp;
                }
            }

            free(rhs);
            free(Vi);
        }
        free(rhs_list_i);
        free(rhs_list_j);

        pSPARC->Eexx /= pSPARC->dV;

    } else {
        int Nstates_occ = pSPARC->Nstates_occ;
        int Nband_bandcomm_M = pSPARC->Nband_bandcomm_M;

        for (int spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor ++) {
            if (Nband_bandcomm_M == 0) continue;
            double *occ = (pSPARC->spin_typ == 1) ? (pSPARC->occ + spinor * Ns) : pSPARC->occ;
            double *Xi_times_psi = (double *) calloc(Nband_bandcomm_M * Nstates_occ, sizeof(double));
            assert(Xi_times_psi != NULL);

            // perform matrix multiplication psi' * X using ScaLAPACK routines
            cblas_dgemm( CblasColMajor, CblasTrans, CblasNoTrans, Nband_bandcomm_M, Nstates_occ, DMnd,
                        1.0, pSPARC->Xorb + spinor * DMnd, DMndsp, pSPARC->Xi + spinor * DMnd, 
                        DMndsp, 0.0, Xi_times_psi, Nband_bandcomm_M);

            if (size > 1) {
                // sum over all processors in dmcomm
                MPI_Allreduce(MPI_IN_PLACE, Xi_times_psi, Nband_bandcomm_M*Nstates_occ, 
                            MPI_DOUBLE, MPI_SUM, comm);
            }

            for (i = 0; i < Nband_bandcomm_M; i++) {
                temp = 0.0;
                for (j = 0; j < Nstates_occ; j++) {
                    temp += Xi_times_psi[i+j*Nband_bandcomm_M] * Xi_times_psi[i+j*Nband_bandcomm_M];
                }
                temp *= occ[i + pSPARC->band_start_indx];
                pSPARC->Eexx += temp;
            }

            free(Xi_times_psi);
        }
    }

    if (pSPARC->npband > 1) {
        MPI_Allreduce(MPI_IN_PLACE, &pSPARC->Eexx, 1, MPI_DOUBLE, MPI_SUM, pSPARC->blacscomm);
    }

    if (pSPARC->npspin > 1) {
        MPI_Allreduce(MPI_IN_PLACE, &pSPARC->Eexx, 1, MPI_DOUBLE, MPI_SUM, pSPARC->spin_bridge_comm);
    }

    pSPARC->Eexx /= (pSPARC->Nspin + 0.0);
    pSPARC->Eexx *= -pSPARC->exx_frac;
}



/**
 * @brief   Solving Poisson's equation using FFT or KRON
 *          
 *          This function only works for solving Poisson's equation with real right hand side
 *          option: 0 - solve poissons equation,       1 - solve with pois_const_stress
 *                  2 - solve with pois_const_stress2, 3 - solve with pois_const_press
 */
void poissonSolve(SPARC_OBJ *pSPARC, double *rhs, double *pois_const, 
                int ncol, int DMnd, int *dims, double *Vi, MPI_Comm comm) 
{    
    int i, k, lsize, lrank, ncolp;
    int *sendcounts, *sdispls, *recvcounts, *rdispls, **DMVertices, *ncolpp;
    int coord_comm[3], gridsizes[3], DNx, DNy, DNz, Nd, Nx, Ny, Nz;
    double *rhs_loc, *Vi_loc, *rhs_loc_order, *Vi_loc_order;
    sendcounts = sdispls = recvcounts = rdispls = ncolpp = NULL;
    rhs_loc = Vi_loc = rhs_loc_order = Vi_loc_order = NULL;
    DMVertices = NULL;

    MPI_Comm_size(comm, &lsize);
    MPI_Comm_rank(comm, &lrank);    
    Nd = pSPARC->Nd;
    Nx = pSPARC->Nx; Ny = pSPARC->Ny; Nz = pSPARC->Nz;     
    ncolp = ncol / lsize + ((lrank < ncol % lsize) ? 1 : 0);
    /********************************************************************/

    if (lsize > 1){
        // variables for RHS storage
        rhs_loc = (double*) malloc(sizeof(double) * ncolp * Nd);
        rhs_loc_order = (double*) malloc(sizeof(double) * Nd * ncolp);

        // number of columns per proc
        ncolpp = (int*) malloc(sizeof(int) * lsize);

        // variables for alltoallv
        sendcounts = (int*) malloc(sizeof(int)*lsize);
        sdispls = (int*) malloc(sizeof(int)*lsize);
        recvcounts = (int*) malloc(sizeof(int)*lsize);
        rdispls = (int*) malloc(sizeof(int)*lsize);
        DMVertices = (int**) malloc(sizeof(int*)*lsize);
        assert(rhs_loc != NULL && rhs_loc_order != NULL && ncolpp != NULL && 
               sendcounts != NULL && sdispls != NULL && recvcounts != NULL && 
               rdispls != NULL && DMVertices!= NULL);

        for (k = 0; k < lsize; k++) {
            DMVertices[k] = (int*) malloc(sizeof(int)*6);
            assert(DMVertices[k] != NULL);
        }
        /********************************************************************/
        
        // separate equations to different processes in the dmcomm or kptcomm_topo                 
        for (i = 0; i < lsize; i++) {
            ncolpp[i] = ncol / lsize + ((i < ncol % lsize) ? 1 : 0);
        }

        // this part of codes copied from parallelization.c
        gridsizes[0] = Nx; gridsizes[1] = Ny; gridsizes[2] = Nz;
        // compute variables required by gatherv and scatterv
        for (i = 0; i < lsize; i++) {
            MPI_Cart_coords(comm, i, 3, coord_comm);
            // find size of distributed domain over comm
            DNx = block_decompose(gridsizes[0], dims[0], coord_comm[0]);
            DNy = block_decompose(gridsizes[1], dims[1], coord_comm[1]);
            DNz = block_decompose(gridsizes[2], dims[2], coord_comm[2]);
            // Here DMVertices [1][3][5] is not the same as they are in parallelization
            DMVertices[i][0] = block_decompose_nstart(gridsizes[0], dims[0], coord_comm[0]);
            DMVertices[i][1] = DNx;                                                                                     // stores number of nodes instead of coordinates of end nodes
            DMVertices[i][2] = block_decompose_nstart(gridsizes[1], dims[1], coord_comm[1]);
            DMVertices[i][3] = DNy;                                                                                     // stores number of nodes instead of coordinates of end nodes
            DMVertices[i][4] = block_decompose_nstart(gridsizes[2], dims[2], coord_comm[2]);
            DMVertices[i][5] = DNz;                                                                                     // stores number of nodes instead of coordinates of end nodes
        }

        sdispls[0] = 0;
        rdispls[0] = 0;
        for (i = 0; i < lsize; i++) {
            sendcounts[i] = ncolpp[i] * DMnd;
            recvcounts[i] = ncolp * DMVertices[i][1] * DMVertices[i][3] * DMVertices[i][5];
            if (i < lsize - 1) {
                sdispls[i+1] = sdispls[i] + sendcounts[i];
                rdispls[i+1] = rdispls[i] + recvcounts[i];
            }
        }
        /********************************************************************/

        MPI_Alltoallv(rhs, sendcounts, sdispls, MPI_DOUBLE, 
                        rhs_loc, recvcounts, rdispls, MPI_DOUBLE, comm);

        // rhs_full needs rearrangement
        block_dp_to_cart((void *) rhs_loc, ncolp, DMVertices, rdispls, lsize, 
                        Nx, Ny, Nd, (void *) rhs_loc_order, sizeof(double));

        free(rhs_loc);
        // variable for local result Vi
        Vi_loc = (double*) malloc(sizeof(double)* Nd * ncolp);
        assert(Vi_loc != NULL);
    } else {
        // if the size of comm is 1, there is no need to scatter and rearrange the results
        rhs_loc_order = rhs;
        Vi_loc = Vi;
    }   
    
    if (pSPARC->ExxMethod == 0) {
        // solve by fft
        pois_fft(pSPARC, rhs_loc_order, pois_const, ncolp, Vi_loc);
    } else {
        // solve by kron
        pois_kron(pSPARC, rhs_loc_order, pois_const, ncolp, Vi_loc);
    }

    if (lsize > 1)  
        free(rhs_loc_order);

    if (lsize > 1) {
        Vi_loc_order = (double*) malloc(sizeof(double)* Nd * ncolp);
        assert(Vi_loc_order != NULL);

        // Vi_loc needs rearrangement
        cart_to_block_dp((void *) Vi_loc, ncolp, DMVertices, lsize, 
                        Nx, Ny, Nd, (void *) Vi_loc_order, sizeof(double));

        MPI_Alltoallv(Vi_loc_order, recvcounts, rdispls, MPI_DOUBLE, 
                    Vi, sendcounts, sdispls, MPI_DOUBLE, comm);

        free(Vi_loc_order);
    }

    /********************************************************************/
    if (lsize > 1){
        free(Vi_loc);
        free(ncolpp);
        free(sendcounts);
        free(sdispls);
        free(recvcounts);
        free(rdispls);
        for (k = 0; k < lsize; k++) 
            free(DMVertices[k]);
        free(DMVertices);
    }
}



/**
 * @brief   Solve Poisson's equation using FFT in Fourier Space
 * 
 * @param rhs               complete RHS of poisson's equations without parallelization. 
 * @param pois_const        constant for solving possion's equations
 * @param ncol              Number of poisson's equations to be solved.
 * @param sol               complete solutions of poisson's equations without parallelization. 
 * Note:                    This function is complete localized. 
 */
void pois_kron(SPARC_OBJ *pSPARC, double *rhs, double *pois_const, int ncol, double *sol)
{
    if (ncol == 0) return;
    int Nd = pSPARC->Nd;
    KRON_LAP* kron_lap = pSPARC->kron_lap_exx;

    if (pSPARC->BC == 2) {
        for (int n = 0; n < ncol; n++) {
            Lap_Kron(kron_lap->Nx, kron_lap->Ny, kron_lap->Nz, kron_lap->Vx, kron_lap->Vy, kron_lap->Vz,
                    rhs + n*Nd, pois_const, sol + n*Nd);
        }
    } else {
        double *rhs_ = (double *) malloc(sizeof(double)*Nd);
        double *d_cor = (double *) malloc(sizeof(double)*Nd);
        assert(rhs_ != NULL && d_cor != NULL);
        int DMVertices[6] = {0, pSPARC->Nx-1, 0, pSPARC->Ny-1, 0, pSPARC->Nz-1};

        for (int n = 0; n < ncol; n++) {
            for (int i = 0; i < Nd; i++) rhs_[i] = -4*M_PI*rhs[i + n*Nd];
            apply_multipole_expansion(pSPARC, pSPARC->MpExp_exx, 
                pSPARC->Nx, pSPARC->Ny, pSPARC->Nz, pSPARC->Nx, pSPARC->Ny, pSPARC->Nz, DMVertices, rhs_, d_cor, MPI_COMM_SELF);
            for (int i = 0; i < Nd; i++) rhs_[i] -= d_cor[i];
            Lap_Kron(kron_lap->Nx, kron_lap->Ny, kron_lap->Nz, kron_lap->Vx, kron_lap->Vy, kron_lap->Vz,
                    rhs_, kron_lap->inv_eig, sol + n*Nd);
        }
        free(rhs_);
        free(d_cor);
    }
}



/**
 * @brief   Solve Poisson's equation using FFT in Fourier Space
 * 
 * @param rhs               complete RHS of poisson's equations without parallelization. 
 * @param pois_const    constant for solving possion's equations
 * @param ncol              Number of poisson's equations to be solved.
 * @param sol               complete solutions of poisson's equations without parallelization. 
 * Note:                    This function is complete localized. 
 */
void pois_fft(SPARC_OBJ *pSPARC, double *rhs, double *pois_const, int ncol, double *sol) {
    if (ncol == 0) return;
    int i, j, Nd, Nx, Ny, Nz, Ndc;
    double _Complex *rhs_bar;

    Nd = pSPARC->Nd;
    Nx = pSPARC->Nx; Ny = pSPARC->Ny; Nz = pSPARC->Nz; 
    Ndc = Nz * Ny * (Nx/2+1);
    rhs_bar = (double _Complex*) malloc(sizeof(double _Complex) * Ndc * ncol);
    assert(rhs_bar != NULL);
    /********************************************************************/

    // FFT
#if defined(USE_MKL)
    MKL_LONG dim_sizes[3] = {Nz, Ny, Nx};
    MKL_LONG strides_out[4] = {0, Ny*(Nx/2+1), Nx/2+1, 1}; 

    for (i = 0; i < ncol; i++)
        MKL_MDFFT_real(rhs + i * Nd, dim_sizes, strides_out, rhs_bar + i * Ndc);
#elif defined(USE_FFTW)
    int dim_sizes[3] = {Nz, Ny, Nx};

    for (i = 0; i < ncol; i++)
        FFTW_MDFFT_real(dim_sizes, rhs + i * Nd, rhs_bar + i * Ndc);
#endif

    // multiplied by alpha
    for (j = 0; j < ncol; j++) {
        for (i = 0; i < Ndc; i++) {
            rhs_bar[i + j*Ndc] = creal(rhs_bar[i + j*Ndc]) * pois_const[i] 
                                + (cimag(rhs_bar[i + j*Ndc]) * pois_const[i]) * I;
        }
    }

    // iFFT
#if defined(USE_MKL)
    for (i = 0; i < ncol; i++)
        MKL_MDiFFT_real(rhs_bar + i * Ndc, dim_sizes, strides_out, sol + i * Nd);
#elif defined(USE_FFTW)
    for (i = 0; i < ncol; i++)
        FFTW_MDiFFT_real(dim_sizes, rhs_bar + i * Ndc, sol + i * Nd);
#endif

    free(rhs_bar);
}

/**
 * @brief   Gather psi_outers in other bandcomms
 *
 *          The default comm is blacscomm
 */
void gather_psi_occ_outer(SPARC_OBJ *pSPARC, double *psi_outer, double *occ_outer) 
{
    int i, grank, lrank, lsize, Ns, DMnd, DMndsp, Nband;

    MPI_Comm_rank(MPI_COMM_WORLD, &grank);
    MPI_Comm_rank(pSPARC->blacscomm, &lrank);
    MPI_Comm_size(pSPARC->blacscomm, &lsize);

    DMnd = pSPARC->Nd_d_dmcomm;
    DMndsp = DMnd * pSPARC->Nspinor_spincomm;
    Nband = pSPARC->Nband_bandcomm;    
    Ns = pSPARC->Nstates;

    int NsNsp = Ns * pSPARC->Nspin_spincomm;
    int shift = pSPARC->band_start_indx * DMndsp;
    // Save orbitals and occupations and to construct exact exchange operator
    copy_mat_blk(sizeof(double), pSPARC->Xorb, DMndsp, DMndsp, Nband, psi_outer+shift, DMndsp);
    gather_blacscomm(pSPARC, DMndsp, Ns, psi_outer);
    
    for (i = 0; i < NsNsp; i++) 
        occ_outer[i] = pSPARC->occ[i];
    /********************************************************************/
    if (pSPARC->flag_kpttopo_dm && pSPARC->ExxAcc == 0) {
        int rank_kptcomm_topo;
        MPI_Comm_rank(pSPARC->kptcomm_topo, &rank_kptcomm_topo);
        if (pSPARC->flag_kpttopo_dm_type == 1) {
            if (!rank_kptcomm_topo)
                MPI_Bcast(occ_outer, NsNsp, MPI_DOUBLE, MPI_ROOT, pSPARC->kpttopo_dmcomm_inter);
            else
                MPI_Bcast(occ_outer, NsNsp, MPI_DOUBLE, MPI_PROC_NULL, pSPARC->kpttopo_dmcomm_inter);
        } else {
            MPI_Bcast(occ_outer, NsNsp, MPI_DOUBLE, 0, pSPARC->kpttopo_dmcomm_inter);
        }
    }
}



/**
 * @brief   Gather orbitals shape vectors across blacscomm
 */
void gather_blacscomm(SPARC_OBJ *pSPARC, int Nrow, int Ncol, double *vec)
{
    if (pSPARC->blacscomm == MPI_COMM_NULL) return;
    int i, grank, lrank, lsize;
    int *recvcounts, *displs, NB;

    MPI_Comm_rank(MPI_COMM_WORLD, &grank);
    MPI_Comm_rank(pSPARC->blacscomm, &lrank);
    MPI_Comm_size(pSPARC->blacscomm, &lsize);    

    if (lsize > 1) {
        recvcounts = (int*) malloc(sizeof(int)* lsize);
        displs = (int*) malloc(sizeof(int)* lsize);
        assert(recvcounts != NULL && displs != NULL);

        // gather all bands, this part of code copied from parallelization.c
        NB = (pSPARC->Nstates - 1) / pSPARC->npband + 1;
        displs[0] = 0;
        for (i = 0; i < lsize; i++){
            recvcounts[i] = (i < (Ncol / NB) ? NB : (i == (Ncol / NB) ? (Ncol % NB) : 0)) * Nrow;
            if (i != (lsize-1))
                displs[i+1] = displs[i] + recvcounts[i];
        }

        MPI_Allgatherv(MPI_IN_PLACE, 1, MPI_DOUBLE, vec, 
            recvcounts, displs, MPI_DOUBLE, pSPARC->blacscomm);   
        
        free(recvcounts);
        free(displs); 
    }
}


/**
 * @brief   Allocate memory space for ACE operator and check its size for each outer loop
 */
void allocate_ACE(SPARC_OBJ *pSPARC) {
    int i, rank, DMnd, DMndsp, Nstates_occ, Ns, spn_i;    

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    Ns = pSPARC->Nstates;
    DMnd = pSPARC->Nd_d_dmcomm;
    DMndsp = DMnd * pSPARC->Nspinor_spincomm;
        
    Nstates_occ = 0;
    for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
        // construct ACE operator only using occupied states
        int Nstates_occ_temp = 0;
        for (i = 0; i < Ns; i++)
            if (pSPARC->occ[i + spn_i*Ns] > 1e-6) Nstates_occ_temp ++;
        pSPARC->Nstates_occ_list[spn_i] = Nstates_occ_temp;
        Nstates_occ = max(Nstates_occ, Nstates_occ_temp);
    }
    Nstates_occ += pSPARC->EeeAceValState;
    Nstates_occ = min(Nstates_occ, pSPARC->Nstates);                      // Ensure Nstates_occ is less or equal to Nstates        
    
    // Note: occupations are only correct in dmcomm.
    MPI_Allreduce(MPI_IN_PLACE, &Nstates_occ, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    
    if (pSPARC->spincomm_index < 0) return;

    // If number of occupied states changed, need to reallocate memory space
    if (Nstates_occ != pSPARC->Nstates_occ) { 
        if (pSPARC->Nstates_occ > 0) {
        #ifdef DEBUG
        if(!rank) 
            printf("\nNumber of occupied states + Extra states changed : %d\n", Nstates_occ);
        #endif  
            free(pSPARC->Xi);
            free(pSPARC->Xi_kptcomm_topo);
            pSPARC->Xi = NULL;
            pSPARC->Xi_kptcomm_topo = NULL;
        } else {
        #ifdef DEBUG
        if(!rank) 
            printf("\nStarts to use %d states to create ACE operator.\n", Nstates_occ);
        #endif  
        }

        // Xi, ACE operator
        pSPARC->Xi = (double *)malloc(DMndsp * Nstates_occ * sizeof(double));
        // Storage of ACE operator in kptcomm_topo
        pSPARC->Xi_kptcomm_topo = 
                (double *)calloc(pSPARC->Nd_d_kptcomm * pSPARC->Nspinor_spincomm * Nstates_occ , sizeof(double));
        assert(pSPARC->Xi != NULL && pSPARC->Xi_kptcomm_topo != NULL);    
        pSPARC->Nstates_occ = Nstates_occ;
    }
    
    if (Nstates_occ < pSPARC->band_start_indx)
        pSPARC->Nband_bandcomm_M = 0;
    else {
        pSPARC->Nband_bandcomm_M = min(pSPARC->Nband_bandcomm, Nstates_occ - pSPARC->band_start_indx);
    }

#if defined(USE_MKL) || defined(USE_SCALAPACK)
    // create SCALAPACK information for ACE operator
    int nprow, npcol, myrow, mycol;
    // get coord of each process in original context
    Cblacs_gridinfo( pSPARC->ictxt_blacs, &nprow, &npcol, &myrow, &mycol );
    int ZERO = 0, mb, nb, llda, info;
        
    mb = max(1, Nstates_occ);
    nb = mb;
    // nb = (pSPARC->Nstates - 1) / pSPARC->npband + 1; // equal to ceil(Nstates/npband), for int only
    // set up descriptor for storage of orbitals in ictxt_blacs (original)
    llda = mb;
    if (pSPARC->bandcomm_index != -1 && pSPARC->dmcomm != MPI_COMM_NULL) {
        descinit_(pSPARC->desc_M, &Nstates_occ, &Nstates_occ,
                &mb, &nb, &ZERO, &ZERO, &pSPARC->ictxt_blacs, &llda, &info);
        pSPARC->nrows_M = numroc_( &Nstates_occ, &mb, &myrow, &ZERO, &nprow);
        pSPARC->ncols_M = numroc_( &Nstates_occ, &nb, &mycol, &ZERO, &npcol);
    } else {
        for (i = 0; i < 9; i++)
            pSPARC->desc_M[i] = -1;
        pSPARC->nrows_M = pSPARC->ncols_M = 0;
    }

    // descriptor for Xi 
    mb = max(1, DMnd);
    nb = (pSPARC->Nstates - 1) / pSPARC->npband + 1; // equal to ceil(Nstates/npband), for int only
    // set up descriptor for storage of orbitals in ictxt_blacs (original)
    llda = max(1, DMndsp);
    if (pSPARC->bandcomm_index != -1 && pSPARC->dmcomm != MPI_COMM_NULL) {
        descinit_(pSPARC->desc_Xi, &DMnd, &Nstates_occ,
                &mb, &nb, &ZERO, &ZERO, &pSPARC->ictxt_blacs, &llda, &info);
    } else {
        for (i = 0; i < 9; i++)
            pSPARC->desc_Xi[i] = -1;
    }
#endif
}



/**
 * @brief   Create ACE operator in dmcomm
 *
 *          Using occupied + extra orbitals to construct the ACE operator 
 *          Due to the symmetry of ACE operator, only half Poisson's 
 *          equations need to be solved.
 */
void ACE_operator(SPARC_OBJ *pSPARC, double *psi, double *occ, double *Xi) 
{
    int i, rank, nproc_dmcomm, Nband_M, DMnd, DMndsp, ONE = 1, Nstates_occ;    
    double *M, t1, t2, *Xi_, *psi_storage1, *psi_storage2, t_comm;
    /******************************************************************************/

    if (pSPARC->spincomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(pSPARC->dmcomm, &nproc_dmcomm);

    Nband_M = pSPARC->Nband_bandcomm_M;
    DMnd = pSPARC->Nd_d_dmcomm;
    DMndsp = DMnd * pSPARC->Nspinor_spincomm;
    Nstates_occ = pSPARC->Nstates_occ;    

    memset(Xi, 0, sizeof(double) * Nstates_occ*DMndsp);
    // if Nband==0 here, Xi_ won't be used anyway
    Xi_ = Xi + pSPARC->band_start_indx * DMndsp;

    int nproc_blacscomm = pSPARC->npband;
    int reps = (nproc_blacscomm == 1) ? 0 : ((nproc_blacscomm - 2) / 2 + 1); // ceil((nproc_blacscomm-1)/2)
    int Nband_max = (pSPARC->Nstates - 1) / pSPARC->npband + 1;

    MPI_Request reqs[2];
    psi_storage1 = psi_storage2 = NULL;
    if (reps > 0) {
        psi_storage1 = (double *) calloc(sizeof(double), DMnd * Nband_max);
        psi_storage2 = (double *) calloc(sizeof(double), DMnd * Nband_max);
    }
    
    t_comm = 0;
    for (int spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor++) {
        // in case of hydrogen 
        if (pSPARC->Nstates_occ_list[min(spinor, pSPARC->Nspin_spincomm-1)] == 0) continue;

        for (int rep = 0; rep <= reps; rep++) {
            if (rep == 0) {
                if (reps > 0) {
                    t1 = MPI_Wtime();
                    // first gather the orbitals in the rotation way
                    if (DMnd != DMndsp) {
                        copy_mat_blk(sizeof(double), psi + spinor*DMnd, DMndsp, DMnd, pSPARC->Nband_bandcomm, psi_storage2, DMnd);
                        transfer_orbitals_blacscomm(pSPARC, psi_storage2, psi_storage1, rep, reqs, sizeof(double));
                    } else {
                        transfer_orbitals_blacscomm(pSPARC, psi, psi_storage1, rep, reqs, sizeof(double));
                    }
                    t2 = MPI_Wtime();
                    t_comm += (t2 - t1);
                }
                // solve poisson's equations 
                double *occ_ = (pSPARC->spin_typ == 1) ? (occ + spinor*pSPARC->Nstates) : occ;
                solve_half_local_poissons_equation_apply2Xi(pSPARC, Nband_M, psi + spinor*DMnd, DMndsp, occ_, Xi_ + spinor*DMnd, DMndsp);
            } else {
                t1 = MPI_Wtime();
                MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
                double *sendbuff = (rep%2==1) ? psi_storage1 : psi_storage2;
                double *recvbuff = (rep%2==1) ? psi_storage2 : psi_storage1;
                if (rep != reps) {
                    // first gather the orbitals in the rotation way
                    transfer_orbitals_blacscomm(pSPARC, sendbuff, recvbuff, rep, reqs, sizeof(double));
                }
                t2 = MPI_Wtime();
                t_comm += (t2 - t1);

                // solve poisson's equations 
                double *occ_ = (pSPARC->spin_typ == 1) ? (occ + spinor*pSPARC->Nstates) : occ;
                solve_allpair_poissons_equation_apply2Xi(pSPARC, Nband_M, psi + spinor*DMnd, DMndsp, sendbuff, DMnd, occ_, Xi + spinor*DMnd, DMndsp, rep);
            }
        }
    }

    #ifdef DEBUG
        if (!rank) printf("transferring orbitals in rotation wise took %.3f ms\n", t_comm*1e3);
    #endif

    if (reps > 0) {
        free(psi_storage1);
        free(psi_storage2);
    }

    // Allreduce is unstable in valgrind test
    if (nproc_blacscomm > 1) {
        MPI_Request req;
        MPI_Status  sta;
        MPI_Iallreduce(MPI_IN_PLACE, Xi, DMndsp*Nstates_occ, MPI_DOUBLE, MPI_SUM, pSPARC->blacscomm, &req);
        MPI_Wait(&req, &sta);
    }

    /******************************************************************************/
    double alpha = 1.0, beta = 0.0;
    int nrows_M = pSPARC->nrows_M;
    int ncols_M = pSPARC->ncols_M;
    M = (double *) malloc(nrows_M * ncols_M * sizeof(double));
    assert(M != NULL);

    t1 = MPI_Wtime();

    for (int spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor ++ ) {
        // in case of hydrogen 
        if (pSPARC->Nstates_occ_list[min(spinor, pSPARC->Nspin_spincomm)] == 0) continue;

        #if defined(USE_MKL) || defined(USE_SCALAPACK)
        // perform matrix multiplication psi' * W using ScaLAPACK routines    
        pdgemm_("T", "N", &Nstates_occ, &Nstates_occ, &DMnd, &alpha, 
                psi + spinor*DMnd, &ONE, &ONE, pSPARC->desc_Xi, Xi_ + spinor*DMnd, 
                &ONE, &ONE, pSPARC->desc_Xi, &beta, M, &ONE, &ONE, 
                pSPARC->desc_M);
        #else // #if defined(USE_MKL) || defined(USE_SCALAPACK)
        // add the implementation without SCALAPACK
        exit(255);
        #endif // #if defined(USE_MKL) || defined(USE_SCALAPACK)

        if (nproc_dmcomm > 1) {
            // sum over all processors in dmcomm
            MPI_Allreduce(MPI_IN_PLACE, M, nrows_M*ncols_M,
                        MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm);
        }
        
        t2 = MPI_Wtime();
        #ifdef DEBUG
        if(!rank && !spinor) printf("rank = %2d, finding M = psi'* W took %.3f ms\n",rank,(t2-t1)*1e3); 
        #endif

        // perform Cholesky Factorization on -M
        // M = chol(-M), upper triangular matrix
        for (i = 0; i < nrows_M*ncols_M; i++) M[i] = -1.0 * M[i];

        t1 = MPI_Wtime();
        int info = 0;
        if (nrows_M*ncols_M > 0) {
            info = LAPACKE_dpotrf (LAPACK_COL_MAJOR, 'U', Nstates_occ, M, Nstates_occ);
        }
        
        t2 = MPI_Wtime();
        #ifdef DEBUG
        if (!rank && !spinor) 
            printf("==Cholesky Factorization: "
                "info = %d, computing Cholesky Factorization using LAPACKE_dpotrf: %.3f ms\n", 
                info, (t2 - t1)*1e3);
        #else
        (void) info; // suppress unused var warning
        #endif

        // Xi = WM^(-1)
        t1 = MPI_Wtime();
        #if defined(USE_MKL) || defined(USE_SCALAPACK)
        pdtrsm_("R", "U", "N", "N", &DMnd, &Nstates_occ, &alpha, 
                    M, &ONE, &ONE, pSPARC->desc_M, 
                    Xi_ + spinor*DMnd, &ONE, &ONE, pSPARC->desc_Xi);
        #else // #if defined(USE_MKL) || defined(USE_SCALAPACK)
        // add the implementation without SCALAPACK
        exit(255);
        #endif // #if defined(USE_MKL) || defined(USE_SCALAPACK)

        t2 = MPI_Wtime();
        #ifdef DEBUG
        if (!rank && !spinor) 
            printf("==Triangular matrix equation: "
                "Solving triangular matrix equation using cblas_dtrsm: %.3f ms\n", (t2 - t1)*1e3);
        #endif
    }

    free(M);

    // gather all columns of Xi
    gather_blacscomm(pSPARC, DMndsp, Nstates_occ, Xi);
}

/**
 * @brief   Solve half of poissons equation locally and apply to Xi
 */
void solve_half_local_poissons_equation_apply2Xi(SPARC_OBJ *pSPARC, int ncol, double *psi, int ldp, double *occ, double *Xi, int ldxi)
{
    int i, j, k, rank, dims[3], Nband, DMnd;
    int *rhs_list_i, *rhs_list_j, num_rhs, count, loop, batch_num_rhs, NL, base;
    double occ_i, occ_j, *rhs, *Vi;

#ifdef DEBUG_EXX
    double t1, t2;
#endif

    /******************************************************************************/

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    DMnd = pSPARC->Nd_d_dmcomm;    
    dims[0] = pSPARC->npNdx; dims[1] = pSPARC->npNdy; dims[2] = pSPARC->npNdz;
    Nband = pSPARC->Nband_bandcomm;
    if (ncol == 0) return;

    rhs_list_i = (int*) calloc(Nband * ncol, sizeof(int)); 
    rhs_list_j = (int*) calloc(Nband * ncol, sizeof(int)); 
    assert(rhs_list_i != NULL && rhs_list_j != NULL);

    // get index for rhs of poissons equations
    count = 0;
    for (j = 0; j < Nband; j++) {
        if (occ[j + pSPARC->band_start_indx] < 1e-6) continue;
        for (i = j; i < ncol; i++) {             
            rhs_list_i[count] = i;
            rhs_list_j[count] = j;
            count ++;
        }
    }
    num_rhs = count;

#ifdef DEBUG_EXX
    t1 = MPI_Wtime();
#endif
    if (num_rhs > 0) {
        batch_num_rhs = pSPARC->ExxMemBatch == 0 ? 
                        num_rhs : pSPARC->ExxMemBatch * pSPARC->npNd;
                        
        NL = (num_rhs - 1) / batch_num_rhs + 1;                                                // number of loops required
        rhs = (double *)malloc(sizeof(double) * DMnd * batch_num_rhs);                         // right hand sides of Poisson's equation
        Vi = (double *)malloc(sizeof(double) * DMnd * batch_num_rhs);                          // the solution for each rhs
        assert(rhs != NULL && Vi != NULL);

        /*************** Solve all Poisson's equation and find M ****************/
        for (loop = 0; loop < NL; loop ++) {
            base = batch_num_rhs*loop;
            for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
                i = rhs_list_i[count];
                j = rhs_list_j[count];
                for (k = 0; k < DMnd; k++) {
                    rhs[k + (count-base)*DMnd] = psi[k + j*ldp] * psi[k + i*ldp];
                }
            }
            
            poissonSolve(pSPARC, rhs, pSPARC->pois_const, count-base, DMnd, dims, Vi, pSPARC->dmcomm);

            for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
                i = rhs_list_i[count];
                j = rhs_list_j[count];
                
                occ_i = occ[i + pSPARC->band_start_indx];
                occ_j = occ[j + pSPARC->band_start_indx];

                for (k = 0; k < DMnd; k++) {
                    Xi[k + i*ldxi] -= occ_j * psi[k + j*ldp] * Vi[k + (count-base)*DMnd] / pSPARC->dV;
                }

                if (i != j && occ_i > 1e-6) {
                    for (k = 0; k < DMnd; k++) {
                        Xi[k + j*ldxi] -= occ_i * psi[k + i*ldp] * Vi[k + (count-base)*DMnd] / pSPARC->dV;
                    }
                }
            }
        }
        free(rhs);
        free(Vi);
    }

    free(rhs_list_i);
    free(rhs_list_j);

#ifdef DEBUG_EXX
    t2 = MPI_Wtime();
    if(!rank) printf("rank = %2d, solving Poisson's equations took %.3f ms\n",rank,(t2-t1)*1e3); 
#endif
}

/**
 * @brief   transfer orbitals in a cyclic rotation way to save memory
 */
void transfer_orbitals_blacscomm(SPARC_OBJ *pSPARC, void *sendbuff, void *recvbuff, int shift, MPI_Request *reqs, int unit_size)
{
    assert(unit_size == sizeof(double) || unit_size == sizeof(double _Complex));
    MPI_Comm blacscomm = pSPARC->blacscomm;
    if (blacscomm == MPI_COMM_NULL) return;
    int size, rank;
    MPI_Comm_size(blacscomm, &size);
    MPI_Comm_rank(blacscomm, &rank);

    int DMnd = pSPARC->Nd_d_dmcomm;    
    int Ns = pSPARC->Nstates;
    int NB = (pSPARC->Nstates - 1) / pSPARC->npband + 1; // this is equal to ceil(Nstates/npband), for int inputs only
    int srank = (rank-shift+size)%size;
    int rrank = (rank-shift-1+size)%size;
    int Nband_send = srank < (Ns / NB) ? NB : (srank == (Ns / NB) ? (Ns % NB) : 0);
    int Nband_recv = rrank < (Ns / NB) ? NB : (rrank == (Ns / NB) ? (Ns % NB) : 0);
    int rneighbor = (rank+1)%size;
    int lneighbor = (rank-1+size)%size;

    if (unit_size == sizeof(double)) {
        MPI_Irecv(recvbuff, DMnd*Nband_recv, MPI_DOUBLE, lneighbor, 111, blacscomm, &reqs[1]);
        MPI_Isend(sendbuff, DMnd*Nband_send, MPI_DOUBLE, rneighbor, 111, blacscomm, &reqs[0]);
    } else {
        MPI_Irecv(recvbuff, DMnd*Nband_recv, MPI_DOUBLE_COMPLEX, lneighbor, 111, blacscomm, &reqs[1]);
        MPI_Isend(sendbuff, DMnd*Nband_send, MPI_DOUBLE_COMPLEX, rneighbor, 111, blacscomm, &reqs[0]);
    }
    
}


/**
 * @brief   Sovle all pair of poissons equations by remote orbitals and apply to Xi
 */
void solve_allpair_poissons_equation_apply2Xi(
    SPARC_OBJ *pSPARC, int ncol, double *psi, int ldp, double *psi_storage, int ldps, double *occ, double *Xi, int ldxi, int shift)
{
    MPI_Comm blacscomm = pSPARC->blacscomm;
    if (blacscomm == MPI_COMM_NULL) return;
    int size, rank, grank;
    MPI_Comm_size(blacscomm, &size);
    MPI_Comm_rank(blacscomm, &rank);
    MPI_Comm_rank(MPI_COMM_WORLD, &grank);

    // when blacscomm is composed of even number of processors
    // second half of them will solve one less rep than the first half to avoid repetition
    int reps = (size - 2) / 2 + 1;
    if (size%2 == 0 && rank >= size/2 && shift == reps) return;

    int i, j, k, nproc_dmcomm, Ns, dims[3], DMnd;
    int *rhs_list_i, *rhs_list_j, num_rhs, count, loop, batch_num_rhs, NL, base;
    double occ_i, occ_j, *rhs, *Vi, *Xi_l, *Xi_r;

#ifdef DEBUG
    double t1, t2;
#endif

    /******************************************************************************/

    if (ncol == 0) return;
    MPI_Comm_size(pSPARC->dmcomm, &nproc_dmcomm);
    DMnd = pSPARC->Nd_d_dmcomm;    
    Ns = pSPARC->Nstates;
    dims[0] = pSPARC->npNdx; dims[1] = pSPARC->npNdy; dims[2] = pSPARC->npNdz;

    int source = (rank-shift+size)%size;
    int NB = (Ns - 1) / pSPARC->npband + 1; // this is equal to ceil(Nstates/npband), for int inputs only
    int Nband_source = source < (Ns / NB) ? NB : (source == (Ns / NB) ? (Ns % NB) : 0);
    int band_start_indx_source = source * NB;

    rhs_list_i = (int*) calloc(Nband_source * ncol, sizeof(int)); 
    rhs_list_j = (int*) calloc(Nband_source * ncol, sizeof(int)); 
    assert(rhs_list_i != NULL && rhs_list_j != NULL);

    // get index for rhs of poissons equations
    count = 0;
    for (j = 0; j < Nband_source; j++) {
        occ_j = occ[j + band_start_indx_source];
        for (i = 0; i < ncol; i++) {     
            occ_i = occ[i + pSPARC->band_start_indx];
            if (occ_j < 1e-6 && (occ_i < 1e-6 || j + band_start_indx_source >= pSPARC->Nstates_occ)) continue;
            rhs_list_i[count] = i;
            rhs_list_j[count] = j;
            count ++;
        }
    }
    num_rhs = count;

    Xi_l = Xi + pSPARC->band_start_indx * ldxi;
    Xi_r = Xi + band_start_indx_source * ldxi;

    #ifdef DEBUG
    t1 = MPI_Wtime();
    #endif
    if (num_rhs > 0) {
        batch_num_rhs = pSPARC->ExxMemBatch == 0 ? 
                        num_rhs : pSPARC->ExxMemBatch * nproc_dmcomm;
                        
        NL = (num_rhs - 1) / batch_num_rhs + 1;                                                // number of loops required
        rhs = (double *)malloc(sizeof(double) * DMnd * batch_num_rhs);                         // right hand sides of Poisson's equation
        Vi = (double *)malloc(sizeof(double) * DMnd * batch_num_rhs);                          // the solution for each rhs
        assert(rhs != NULL && Vi != NULL);

        /*************** Solve all Poisson's equation and find M ****************/
        for (loop = 0; loop < NL; loop ++) {
            base = batch_num_rhs*loop;
            for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
                i = rhs_list_i[count];
                j = rhs_list_j[count];
                for (k = 0; k < DMnd; k++) {
                    rhs[k + (count-base)*DMnd] = psi_storage[k + j*ldps] * psi[k + i*ldp];
                }
            }
            
            poissonSolve(pSPARC, rhs, pSPARC->pois_const, count-base, DMnd, dims, Vi, pSPARC->dmcomm);
            
            for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
                i = rhs_list_i[count];
                j = rhs_list_j[count];
                
                occ_i = occ[i + pSPARC->band_start_indx];
                occ_j = occ[j + band_start_indx_source];

                if (occ_j > 1e-6) {
                    for (k = 0; k < DMnd; k++) {
                        Xi_l[k + i*ldxi] -= occ_j * psi_storage[k + j*ldps] * Vi[k + (count-base)*DMnd] / pSPARC->dV;
                    }
                }

                if (occ_i > 1e-6) {
                    for (k = 0; k < DMnd; k++) {
                        Xi_r[k + j*ldxi] -= occ_i * psi[k + i*ldp] * Vi[k + (count-base)*DMnd] / pSPARC->dV;
                    }
                }
            }
        }
        free(rhs);
        free(Vi);
    }

    free(rhs_list_i);
    free(rhs_list_j);
    
    #ifdef DEBUG
    t2 = MPI_Wtime();
    if(!grank) printf("rank = %2d, solving Poisson's equations took %.3f ms\n",grank,(t2-t1)*1e3); 
    #endif
}