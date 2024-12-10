/***
 * @file    sqProperties.c
 * @brief   This file contains the functions for force calculation using SQ method.
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
/** BLAS and LAPACK routines */
#ifdef USE_MKL
    #include <mkl.h>
#else
    #include <cblas.h>
    #include <lapacke.h>
#endif

#include "sq.h"
#include "sqDensity.h"
#include "occupation.h"
#include "tools.h"
#include "sqExactExchange.h"


#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))
#define SIGN(a, b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

#define TEMP_TOL (1e-14)


/**
 * @brief   Calculate electron density using SQ method
 */
void Calculate_elecDens_SQ(SPARC_OBJ *pSPARC, int SCFcount) {
    // Gauss Quadrature for electron density
    GaussQuadrature(pSPARC, SCFcount);
}


/**
 * @brief   Compute column of density matrix using Gauss Quadrature
 */
void Gauss_density_matrix_col(SPARC_OBJ *pSPARC, int Nd, int npl, double *DMcol, double *V, double *w, double *D) 
{
    double *wte1 = (double *) calloc(sizeof(double), npl);
    double *gdwte1 = (double *) calloc(sizeof(double), npl);
    double *wgdwte1 = (double *) calloc(sizeof(double), npl);

    for (int j = 0; j < npl; j++) {
        wte1[j] = w[j * npl];
    }

    for (int j = 0; j < npl; j++) {
        gdwte1[j] = wte1[j] * smearing_function(
                    pSPARC->Beta, D[j], pSPARC->Efermi, pSPARC->elec_T_type);
    }

    cblas_dgemv (CblasColMajor, CblasNoTrans, npl, npl, 1.0, 
                    w, npl, gdwte1, 1, 0.0, wgdwte1, 1);

    cblas_dgemv (CblasColMajor, CblasNoTrans, Nd, npl, 1.0/pSPARC->dV, 
                    V, Nd, wgdwte1, 1, 0.0, DMcol, 1);
        
    free(wte1);
    free(gdwte1);
    free(wgdwte1);
}

/**
 * @brief   Compute all columns of density matrix using Gauss Quadrature
 */
void calculate_density_matrix_SQ(SPARC_OBJ *pSPARC)
{
    SQ_OBJ *pSQ = pSPARC->pSQ; 
    if (pSQ->dmcomm_SQ == MPI_COMM_NULL) return;

    int DMnd = pSQ->DMnd_SQ;
    int *nloc = pSQ->nloc;
    int Nx_loc = pSQ->Nx_loc;
    int Ny_loc = pSQ->Ny_loc;
    int Nd_loc = pSQ->Nd_loc;
    int NxNy_loc = Nx_loc*Ny_loc;
    int center = nloc[0] + nloc[1]*Nx_loc + nloc[2]*NxNy_loc;
    int flag_exxPot = (pSPARC->usefock > 0) && (pSPARC->usefock % 2 == 0) 
                   && (pSPARC->ExxAcc == 1) && (pSPARC->SQ_gauss_hybrid_mem == 0);

    for (int nd = 0; nd < DMnd; nd++) {
        if (pSPARC->SQ_gauss_mem == 0) {
            double *t0 = (double *) calloc(sizeof(double), Nd_loc);
            t0[center] = 1;

            // calculate exx potential if not saved
            if (flag_exxPot == 1) {                
                double t1 = MPI_Wtime();
                compute_exx_potential_node_SQ(pSPARC, nd, pSQ->exxPot[0]);
                double t2 = MPI_Wtime();
                pSPARC->ACEtime += (t2 - t1);
            }

            double lambda_min, lambda_max;
            LanczosAlgorithm_gauss(pSPARC, t0, &lambda_min, &lambda_max, nd);
            Gauss_density_matrix_col(pSPARC, pSQ->Nd_loc, pSPARC->SQ_npl_g, pSQ->Dn[nd], pSQ->lanczos_vec, pSQ->w, pSQ->gnd[nd]);
            free(t0);
        } else {
            // Already saved
            Gauss_density_matrix_col(pSPARC, pSQ->Nd_loc, pSPARC->SQ_npl_g, pSQ->Dn[nd], pSQ->lanczos_vec_all[nd], pSQ->w_all[nd], pSQ->gnd[nd]);
        }
    }
}