#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <time.h>
#ifdef USE_MKL
    #define MKL_Complex16 double _Complex
    #include <mkl.h>
#else
    #include <cblas.h>
    #include <lapacke.h>
#endif

#include "tools_mlff.h"
#include "spherical_harmonics.h"
#include "soap_descriptor.h"
#include "mlff_types.h"
#include "linearsys.h"
#include "isddft.h"
#include "tools.h"
#include "ddbp_tools.h"


#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))

/*
SOAP_CUR_sparsify function computes the IDs of local descriptors which can be removed from the training dataset
 ** All the descriptors here corresponds to same element type
[Input]
1. X2: 2-body local descriptors in the training dataset
2. X3: 2-body local descriptors in the training dataset
3. n_descriptor: number of descriptors in the training datset
4. size_X2: length of 2-body descriptors
5. size_X2: length of 3-body descriptors
6. beta_2, beta_3, xi_3: parameters in SOAP kernel
[Output]
1. K: Gram matrix
2. highrank_ID_descriptors: ID's of high rank descriptors
*/

void SOAP_CUR_sparsify(int kernel_typ, double **X2, double **X3, int n_descriptor, int size_X2, int size_X3, double beta_2, double beta_3, double xi_3, dyArray *highrank_ID_descriptors, int N_low_min){
	
	int count=0, info, i, j;
	double w[n_descriptor], w1[n_descriptor], *K;
	K = (double *) malloc(n_descriptor*n_descriptor*sizeof(double));
	for (i = 0; i < n_descriptor; i++){
		w1[i] = 0.0;
		for (j = 0; j < n_descriptor; j++){
			K[count] = 0.0;
			count++;
		}
	}

	for (int i=0; i < n_descriptor; i++){
		for (int j = 0; j < n_descriptor; j++){
				K[i*n_descriptor + j] = soap_kernel(kernel_typ, X2[i],  X3[i],  X2[j],  X3[j],
			 							beta_2, beta_3, xi_3, size_X2, size_X3);
		}
	}


	// FILE* fid;
	// fid = fopen('Ktemp.txt','w');
	// for (int i=0; i < n_descriptor; i++){
	// 	for (int j = 0; j < n_descriptor; j++){
	// 			fprintf(fid,"%10.9f ",K[i*n_descriptor + j]);
	// 	}
	// 	fprintf(fid,"\n");
	// }
	// fclose(fid);



	info = LAPACKE_dsyevd( LAPACK_ROW_MAJOR, 'V', 'U', n_descriptor, K, n_descriptor, w );
    /* Check for convergence */
    if( info > 0 ) {
            printf( "LAPACKE_dsyev in SOAP_gram_matrix failed to compute eigenvalues.\n" );
            exit( 1 );
    }
    int N_low =0;
    for (i = 0; i < n_descriptor; i++){
    	if (w[i] < 0.0000000001){
    		N_low += 1;
    		// append_dyarray(highrank_ID_descriptors, i);
    	}
    }

    N_low = max(N_low_min, N_low);

    for (int i= 0; i < n_descriptor; i++){
    	for (int j = 0; j < n_descriptor; j++){
    		if (w[j] < 0.0000000001){
    			w1[i] += (1.0/N_low) * K[i*n_descriptor+j] * K[i*n_descriptor+j];
    		}
    	}

    }
    double temp;
    int temp_idx;


    for (int i = 0; i < n_descriptor - N_low; i++){
    	temp = smallest(w1, n_descriptor);
    	temp_idx = lin_search_double(w1, n_descriptor, temp);
    	w1[temp_idx] = 10000000000;
    	append_dyarray(highrank_ID_descriptors, temp_idx);
    	// printf("N_low: %d, n_descriptor: %d, highrank_ID_descriptors->len: %d\n",N_low,n_descriptor,highrank_ID_descriptors->len);
    }
    free(K);

}
