#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <mpi.h>
/* BLAS and LAPACK routines */
#ifdef USE_MKL
    #include <mkl.h>
#else
    #include <cblas.h>
    #include <lapacke.h>
#endif
/* ScaLAPACK routines */
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

#include "eigenSolver.h"
#include "tools.h" 
#include "linearSolver.h" // Lanczos
#include "lapVecRoutines.h"
#include "hamiltonianVecRoutines.h"
#include "occupation.h"
#include "isddft.h"
#include "parallelization.h"
#include "sq3.h"
#include "cs.h"
#include "linearAlgebra.h"
#include "ddbp.h"
#include "extFPMD.h"
#include "cyclix_tools.h"
#include "ofdft.h"
#include "krylovschur.h"
#include "inverse.h"

#define TEMP_TOL 1e-12

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))




void krylovschur_min(SPARC_OBJ *pSPARC, int *DMVertices,
             double *eigmin, double *eigvec_min, double *x0, double TOL,
             int MAXIT, int nmax, int nmin, int nwanted) 
{	
	MPI_Comm comm;
    comm = pSPARC->dmcomm_phi;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int DMnd = pSPARC->Nd_d;
	int n = DMnd;
	int iteration;
	int info;
	int size_T, size_M;
	double norm_x0;
	int nc, converged;

	double *T, *U, *d, *e, *Q, *D, *zz, *M, *zz_min, *u, *v, *d_hess, *e_hess, *tau, *w, *M_temp;
	double alpha, beta, temp;

	// initialize T and U (column major)
	// T on each processor
	// U size dependent on domain size (n) of the processor
	T = (double *) malloc(sizeof(double)*(nmax+1)*(nmax+1));
	U = (double *) malloc(sizeof(double)*n*(nmax+1));

	VectorDotProduct(x0, x0, DMnd, &norm_x0, comm);

	for (int i = 0; i < n; i++){
		U[i] = x0[i]/ sqrt(norm_x0);
	}

	int nstrt = 0;
	int nconv = 0;

	// FILE *fp;
	// if(!rank)
	// 	fp = fopen("yy.txt","w");

	for (int iter = 0; iter < MAXIT; iter++){

		iteration = iter;
		symLanczos(pSPARC, nstrt, nmax, T, U);		

		// if (!rank){
		// 	for (int mm = 0; mm < n; mm++){
		// 		for (int nn = 0; nn < nmax+1; nn++){
		// 			fprintf(fp,"%.14f ",U[nn*n+mm]);
		// 		}
		// 		fprintf(fp,"\n");
		// 	}
		// 	fclose(fp);
		// }
		// MPI_Barrier(MPI_COMM_WORLD);
		// exit(31);

		size_T = nmax-nconv;

		d = (double *) malloc(sizeof(double)*size_T*1);
		e = (double *) malloc(sizeof(double)*(size_T-1)*1);
		for (int jj = 0; jj < size_T; jj++){
			d[jj] = T[(jj+nconv)*(nmax+1)+(jj+nconv)];
		}
		for (int jj = 0; jj < size_T-1; jj++){
			e[jj] = T[(jj+nconv+1)*(nmax+1)+(jj+nconv)];
		}

		Q = (double *) malloc(sizeof(double)*size_T*size_T);

		info = LAPACKE_dsteqr( LAPACK_COL_MAJOR, 'I', size_T, d, e, Q, size_T );
		if (info != 0){
			printf("LAPACKE_dsteqr failed inside krylovschur_min\n");
			exit(1);
		}		

		zz = (double *)malloc(sizeof(double)*size_T);
		for (int jj = 0; jj < size_T; jj++){
			zz[jj] = T[(nmax-1)*(nmax+1)+nmax] * Q[jj*size_T+nmax-nconv-1];
		}		

		matmatprod(&U[(nconv)*n], &Q[0], &U[(nconv)*n], n, nmax-nconv, nmin-nconv);

		// checking for convergence
		nc = nconv;
		for (int ii =nconv; ii < nmin; ii++){
			if (fabs(zz[ii-nconv]) < TOL) {
				nc = nc +1;
				converged = nc;
			} else {
				break;
			}
		}

		if (nc > nconv){

			for (int jj = 0; jj < nc-nconv; jj++){
				T[(jj+nconv)*(nmax+1)+jj+nconv] = d[jj];
			}

			T[(nc)*(nmax+1)+nc-1] = 0.0;
			T[(nc-1)*(nmax+1)+nc] = 0.0;

			
		}

		size_M = nmin-nc;
		M = (double *) calloc(size_M*size_M,sizeof(double));
		zz_min = (double *) malloc(sizeof(double)*size_M);

		for (int jj = 0; jj < size_M; jj++){
			zz_min[jj] = zz[nc-nconv+jj];
			M[jj*size_M+jj] = d[nc-nconv+jj];
		}

		

		u = (double *) malloc(sizeof(double)*size_M);

		reflector(zz, size_M, u, &alpha, &beta);

		// if (iter == 1){
		// 	fprintf(fp,"%.14f\n",alpha);
		// 	fprintf(fp,"%.14f\n",beta);
		// 	// for (int i = 0; i < size_M; i++){
		// 	// 	fprintf(fp, "%.14f\n",u[i] );
		// 	// }

		// 	// for (int i = 0; i < size_M; i++){
		// 	// 	for (int j = 0; j < size_M; j++){
		// 	// 		fprintf(fp, "%.14f ",M[j*size_M+i] );
		// 	// 	}
		// 	// 	fprintf(fp,"\n");
		// 	// }
		// 	// fclose(fp);
		// 	exit(4);
		// }

		v = (double *) malloc(sizeof(double)*size_M);
		for (int jj = 0; jj < size_M; jj++){
			v[jj] = beta * u[jj];
		}

		w = (double *) calloc(n, sizeof(double));

		for (int jj = 0; jj < nmin-nc; jj++){
			for (int kk = 0; kk < n; kk++){
				w[kk] = w[kk] + U[(nc+jj)*n + kk]*u[jj];
			}
		}	

		for (int jj = 0; jj < nmin-nc; jj++){
			for (int kk = 0; kk < n; kk++){
				U[(nc+jj)*n + kk] = U[(nc+jj)*n + kk] - w[kk]*v[jj];
			}
		}


		free(w);
		w = (double *) calloc(size_M, sizeof(double));

		for (int jj = 0; jj < size_M; jj++){
			for (int kk = 0; kk < size_M; kk++){
				w[kk] += M[jj*size_M + kk]*u[jj];
			}
		}

		temp = norm_vector_serial(u, w, size_M);
		temp = temp*temp;


		for (int jj = 0; jj < size_M; jj++){
			for (int kk = 0; kk < size_M; kk++){
				// VectorDotProduct(u, w, size_W, &temp, comm)
				M[kk*size_M+jj] = M[kk*size_M+jj] - w[jj]*v[kk] - v[jj]*w[kk] + temp*v[jj]*v[kk];
			}
		}

		M_temp = (double *) malloc(sizeof(double)*size_M*size_M);

		for (int jj = 0; jj < size_M; jj++){
			for (int kk = 0; kk < size_M; kk++){
				// VectorDotProduct(u, w, size_W, &temp, comm)
				M_temp[jj*size_M+kk] = 0.5*(M[jj*size_M+kk]+M[kk*size_M+jj]) ;
			}
		}

		for (int jj = 0; jj < size_M; jj++){
			for (int kk = 0; kk < size_M; kk++){
				M[jj*size_M+kk] = M_temp[jj*size_M+kk] ;
			}
		}

		d_hess = (double *) malloc(sizeof(double)*size_M);
		e_hess = (double *) malloc(sizeof(double)*(size_M-1));
		tau = (double *) malloc(sizeof(double)*(size_M));

		LAPACKE_dsytrd(LAPACK_COL_MAJOR, 'U', size_M, M_temp, size_M, d_hess, e_hess, tau);
		LAPACKE_dorgtr(LAPACK_COL_MAJOR, 'U', size_M, M_temp, size_M, tau);
		// M_temp is Q and tridigonal form of M_temp is stored in d_hess and e_hess

	

		matmatprod(&U[(nc)*n], &M_temp[0], &U[(nc)*n], n, nmin-nc, nmin-nc);

		for (int jj = 0; jj < n; jj++){
			U[n*nmin+jj] = U[n*nmax+jj];
		}

		// for (int jj = 0; jj < nmin-nc; jj++){
		// 	for (kk = 0; kk < nmin-nc; kk++){
		// 		T[(nc+kk)*(nmin-nc)+jj] = M[kk*(nmin-nc)+jj];
		// 	}
		// }

		for (int jj = 0; jj < nmin-nc; jj++){
			T[(jj+nc)*(nmax+1)+(jj+nc)] = d_hess[jj];
		}

		for (int jj = 0; jj < nmin-nc -1; jj++){
			T[(jj+nc+1)*(nmax+1)+(jj+nc)] = e_hess[jj];
		}

		for (int jj = 0; jj < nmin-nc-1; jj++){
			T[(jj+nc)*(nmax+1)+(jj+nc+1)] = e_hess[jj];
		}

		T[(nmin-1)*(nmax+1)+nmin] = alpha;
		T[(nmin)*(nmax+1)+nmin-1] = alpha;


		nconv = nc;

		free(d);
		free(e);
		free(Q);
		free(zz);
		free(M);
		free(zz_min);
		free(u);
		free(v);
		free(w);
		free(M_temp);
		free(d_hess);
		free(e_hess);
		free(tau);

		if (nconv >= nwanted){
			break;
		}
		nstrt = nmin;		
	}

	for (int i = 0; i < nconv; i++){
		eigmin[i] = T[i*(nmax+1)+i];
	}
	for (int i = 0; i < nconv; i++){
		for (int j = 0; j < n; j++){
			eigvec_min[i*n + j] = U[i*n + j];
		}
	}

	if (!rank){
		printf("Krylov-Schur took %d iterations to converge!\n", iteration);
	}

	free(T);
	free(U);
}


void reflector(double *x, int m, double *u, double *alpha, double *beta)
{
	double scale = norm_vector(x,m);
	double phase;

	if (scale ==0){
		*beta =0.0;
		*alpha = 0.0;
		for (int i =0; i < m; i++){
			u[i] = x[i];
		}
	} else {
		for (int i =0; i < m; i++){
			u[i] = x[i]/scale;
		}
		if (u[m-1] != 0){
			phase = u[m-1]/fabs(u[m-1]);
			for (int i =0; i < m; i++){
				u[i] = u[i]*phase;
			}
		} else {
			phase = 1.0;
		}

		u[m-1] += 1.0;
		*beta = 1.0/u[m-1];
		*alpha = -1.0*scale*phase;
		
	}
}

double norm_vector(double *x, int n){
	double norm_x = 0.0;

	for (int i = 0; i < n; i++){
		norm_x += x[i]*x[i];
	}
	norm_x = sqrt(norm_x);
	return norm_x;

}

void symLanczos(SPARC_OBJ *pSPARC, int nstrt, int nend, double *T, double *U)
{	
	int DMnd = pSPARC->Nd_d;
	int n = DMnd;
	int *DMVertices;
	DMVertices = pSPARC->DMVertices;
	MPI_Comm comm;
    comm = pSPARC->dmcomm_phi;
	double *delta;
	delta = (double *) malloc(sizeof(double)*(nend+1));




	for (int jj = nstrt; jj < nend; jj++){
		HamiltonianVecRoutines_OFDFT_inverse(pSPARC, DMnd, DMVertices, &U[jj*n], &U[(jj+1)*n], comm);

		
		VectorDotProduct(&U[jj*n], &U[(jj+1)*n], n, &T[jj*(nend+1)+jj], comm);
		for (int kk = 0; kk < n; kk++){
			U[(jj+1)*n + kk] = U[(jj+1)*n + kk] -  U[(jj)*n + kk] * T[jj*(nend+1)+jj];
		}

		if (jj > 0){
			for (int kk = 0; kk < n; kk++){
				U[(jj+1)*n + kk] = U[(jj+1)*n + kk] - U[(jj-1)*n + kk] * T[jj*(nend+1)+jj-1];
			}
		}

		

		for (int kk = 0; kk < jj+1; kk++){
			VectorDotProduct(&U[kk*n], &U[(jj+1)*n], n, &delta[kk], comm);
		}

		for (int kk = 0; kk < jj+1; kk++){
			for (int ll = 0; ll < n; ll++){
				U[(jj+1)*n+ll] = U[(jj+1)*n+ll] - U[kk*n+ll]*delta[kk];
			}
		}

		VectorDotProduct(&U[(jj+1)*n], &U[(jj+1)*n], n, &T[(jj+1)*(nend+1)+jj], comm);

		T[(jj+1)*(nend+1)+jj] = sqrt(T[(jj+1)*(nend+1)+jj]);
		T[jj*(nend+1)+jj+1] = T[(jj+1)*(nend+1)+jj];

		

		if (T[(jj+1)*(nend+1)+jj] == 0){
			printf("Error due to zero diagonal\n");
			exit(1);
		} else {
			for (int kk = 0; kk < n; kk++){
				U[(jj+1)*n + kk] =  U[(jj+1)*n + kk]/T[(jj+1)*(nend+1)+jj];
			}
		}

		



	}
	free(delta);
}

void matmatprod(double *A, double *B, double *C, int n1, int n2, int n3){

	double *C1;
	C1 = (double *) malloc(sizeof(double)*n1*n3);
	for (int i = 0; i < n1*n3; i++){
		C1[i] = 0.0;
	}

	for (int i = 0; i < n1; i++){
		for (int j = 0; j < n2; j++){
			for (int k = 0; k < n3; k++){
				C1[k*n1 + i] += A[j*n1+i]*B[k*n2+j];
			}
		}
	}

	for (int i = 0; i < n1*n3; i++){
		C[i] = C1[i];
	}
	free(C1);
}

double norm_vector_serial(double *a, double *b, int N)
{
	double norm = 0.0;

	for (int i = 0; i < N; i++){
		norm += a[i]*b[i];
	}

	norm = sqrt(norm);

	return norm;


}
