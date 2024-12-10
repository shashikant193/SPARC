/**
 * @file    cs.h
 * @brief   This file contains the function declarations for the complementary subspace method.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 *          Hua Huang <huangh223@gatech.edu>
 *          Edmond Chow <echow@cc.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef CS_H
#define CS_H 

#include "isddft.h"


/**
 * @brief Find the bounds for the dense Chebyshev filtering on the subspace 
 *        Hamiltonian.
 *
 * @param lambda_cutoff  The cutoff parameter used for the original Hamiltonian.
 * @param eigmin         Minimum eigenvalue of the original Hamiltonian.
 * @param eigmin_calc    The previous minimum calculated eigenvalue of the 
 *                       subspace Hamiltonian, not referenced if isFirstIt = 0.
 * @param isFirstIt      Flag to check if this is the first Chebyshev iteration.
 */
void Chebyshevfilter_dense_constants(
  const SPARC_OBJ *pSPARC, const double lambda_cutoff, const double eigmin, 
  const double eigmin_calc, const int isFirstIt, double *a, double *b, double *a0);



/**
 * @brief Find Y = (Hs + cI) * X, where X is distributed column-wisely.
 *
 * @param Hs   Symmetric dense matrix, all proc have a copy of the full matrix.
 * @param c    A constant shift.
 * @param X    A bunch of vectors, to be multiplied.
 * @param ncol Number of columns of X distributed in current process. 
 */
void mat_vectors_mult(
  SPARC_OBJ *pSPARC, const int N, const double *Hs, const int ncol, 
  const double c, const double *X, double *Y, const MPI_Comm comm);

/**
 * @brief   Initialize CS emthod.
 */
void init_CS(SPARC_OBJ *pSPARC);

/**
 * @brief   Initialize complementary subspace eigensolver.
 */
void init_CS_CheFSI(SPARC_OBJ *pSPARC);

/**
 * @brief   Free complementary subspace eigensolver.
 */
void free_CS(SPARC_OBJ *pSPARC);

/**
 * @brief   Chebyshev-filtered subspace iteration eigensolver.
 */
void CheFSI_dense_eig(
  SPARC_OBJ *pSPARC, int Ns, int Nt, double lambda_cutoff, int repeat_chefsi, 
  int count, int k, int spn_i);


/**
 * @brief   Lanczos algorithm for calculating min and max eigenvalues
 *          for a symmetric dense matrix.
 */
void Lanczos_dense_seq(
  SPARC_OBJ *pSPARC, const double *A, const int N, 
  double *eigmin, double *eigmax, double *x0, const double TOL_min, 
  const double TOL_max, const int MAXIT, int k, int spn_i, MPI_Comm comm);


/**
 * @brief Perform Chebyshev filtering.
 *        Y = Pm(Hs) X = Tm((Hs - c)/e) X, where Tm is the Chebyshev polynomial 
 *        of the first kind, c = (a+b)/2, e = (b-a)/2.
 *
 * @param N  Matrix size of Hs.
 * @param m  Chebyshev polynomial degree.
 * @param a  Filter bound. a -> -1.
 * @param b  Filter bound. b -> +1.
 * @param a0 Filter scaling factor, Pm(a0) = 1.
 */
void ChebyshevFiltering_dense(
  SPARC_OBJ *pSPARC, const double *Hs, const int N, double *X, double *Y, 
  const int ncol, const int m, const double a, const double b, const double a0, 
  const int k, const int spn_i, const MPI_Comm comm, double *time_info);


/**
 * @brief   Solve subspace eigenproblem Hp * x = lambda * x for the top Nt 
 *          eigenvalues/eigenvectors using the CheFSI algorithm.
 *
 *          Note: Hp = Psi' * H * Psi, where Psi' * Psi = I. 
 *          
 */
void Solve_partial_standard_EigenProblem_CheFSI(
  SPARC_OBJ *pSPARC, double lambda_cutoff, int Nt, int repeat_chefsi, int k, 
  int count, int spn_i);


/**
 * @brief   Solve subspace eigenproblem Hp * x = lambda * x for the top Nt eigenvalues/eigenvectors.
 *
 *          Note: Hp = Psi' * H * Psi, where Psi' * Psi = I. 
 *          
 *          TODO: At some point it is better to use ELPA (https://elpa.mpcdf.mpg.de/) 
 *                for solving subspace eigenvalue problem, which can provide up to 
 *                3x speedup.
 */
void Solve_partial_standard_EigenProblem(SPARC_OBJ *pSPARC, int Nt, int k, int spn_i);

void Subspace_partial_Rotation_CS(SPARC_OBJ* pSPARC, int k, int spn_i);

/**
 * @brief   Perform subspace rotation, i.e. rotate the orbitals, for the top Nt states.
 *
 *          This is just to perform a matrix-matrix multiplication: Psi = Psi * Qt.
 *          Note that Psi, Q and PsiQ are distributed block cyclically, Psi_rot is
 *          the band + domain parallelization format of PsiQ.
 */
void Subspace_partial_Rotation(SPARC_OBJ *pSPARC, double *Psi, double *Q, double *PsiQ, 
                                double *Psi_rot, int Nt, int k, int spn_i);

void update_rho_cs(SPARC_OBJ *pSPARC, double *rho);

/**
 * @brief   Calculate band structure energy when Complementary subspace method 
 *          is turned on.
 */
double Calculate_Eband_CS(SPARC_OBJ *pSPARC);

#ifdef USE_DP_SUBEIG
/**
 * @brief   Perform subspace partial rotation, i.e. rotate the top Nt orbitals, using domain 
 *          parallelization data partitioning. 
 *
 *          This is just to perform a matrix-matrix multiplication: PsiQ = Psi* Q(:,Ns-Nt+1:Ns), here 
 *          Psi == Y, Q == eigvec. Note that Y and YQ are in domain parallelization data 
 *          layout. We use MPI_Alltoallv to convert the obtained YQ back to the band + domain 
 *          parallelization format in SPARC and copy the transformed YQ to Psi_rot. 
 */
void DP_Subspace_partial_Rotation(SPARC_OBJ *pSPARC, double *Psi_rot, int Nt);

#endif 


void Calculate_nonlocal_forces_linear_CS(SPARC_OBJ *pSPARC);

void nonlocal_forces_intPsiTChi(SPARC_OBJ *pSPARC, double *alpha, double *Psi, int DMnd, int ncol, int n_start, int n_end);

void nonlocal_forces_intChiTdPsi(SPARC_OBJ *pSPARC, double *alpha, double *Psi, int DMnd, int ncol, int n_start, int n_end);


#endif // EIGENSOLVER_H 
