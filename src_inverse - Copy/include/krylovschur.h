#ifndef KRYLOVSCHUR_H
#define KRYLOVSCHUR_H 

#include "isddft.h"
void krylovschur_min(SPARC_OBJ *pSPARC, int *DMVertices,
             double *eigmin, double *eigvec_min, double *x0, double TOL,
             int MAXIT, int nmax, int nmin, int nwanted);
void reflector(double *x, int m, double *u, double *alpha, double *beta);
double norm_vector(double *x, int n);
void symLanczos(SPARC_OBJ *pSPARC, int nstrt, int nend, double *T, double *U);
void matmatprod(double *A, double *B, double *C, int n1, int n2, int n3);
double norm_vector_serial(double *a, double *b, int N);

#endif //KRYLOVSCHUR_H
