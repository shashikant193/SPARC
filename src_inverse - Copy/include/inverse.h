#ifndef INVERSE_H
#define INVERSE_H

#include "isddft.h"

void main_INVERSE(SPARC_OBJ *pSPARC);
void HamiltonianVecRoutines_OFDFT_inverse(
        SPARC_OBJ *pSPARC, int DMnd, int *DMVertices,
        double *u, double *Hu, MPI_Comm comm) ;

void linear_system_Ax_inverse(SPARC_OBJ *pSPARC, int DMnd, int *DMVertices, double epsilon, double *u,
        double *x, double *Ax, MPI_Comm comm);

void Calculate_Inversion_derivative(SPARC_OBJ *pSPARC);
void LBFGS_inverse(SPARC_OBJ *pSPARC);
void CG_inverse(SPARC_OBJ *pSPARC, 
    int DMnd, int *DMVertices, double epsilon, double *u, double *x0, double *b, double *x, int max_iter, double tol,  MPI_Comm comm
);



#endif // INVERSE_H