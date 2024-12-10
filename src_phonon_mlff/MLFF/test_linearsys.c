#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include "mkl.h"
#include "tools_mlff.h"
#include "spherical_harmonics.h"
#include "soap_descriptor.h"
#include "mlff_types.h"
#include "sparsification.h"
#include "regression.h"
#include "linearsys.h"

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))

int main() {

	double *X2_str, *X3_str, *X2_tr, *X3_tr, *dX2_str, *dX3_str, beta_2, beta_3, xi_3, y, dy;
	int size_X2, size_X3;

	size_X2 = 10;
	size_X3 = 20;
	xi_3 = 4;
	beta_2 = 0.5;
	beta_3 = 0.5;

	X2_str = (double *) malloc(sizeof(double)*size_X2);
	X2_tr = (double *) malloc(sizeof(double)*size_X2);
	dX2_str = (double *) malloc(sizeof(double)*size_X2);
	X3_str = (double *) malloc(sizeof(double)*size_X3);
	X3_tr = (double *) malloc(sizeof(double)*size_X3);
	dX3_str = (double *) malloc(sizeof(double)*size_X3);
	printf("X2\n");
	for (int i = 0; i < size_X2; i++){
		X2_str[i] = i+0.1;
		X2_tr[i] = 2*i+0.32;
		dX2_str[i] = 2*i*i+0.32;
		printf("%f %f %f\n",X2_str[i], X2_tr[i], dX2_str[i]);
	}
	printf("X3\n");
	for (int i = 0; i < size_X3; i++){
		X3_str[i] = i+0.134;
		X3_tr[i] = 2*i+0.3672;
		dX3_str[i] = 2*i*i*i+0.3672;
		printf("%f %f %f\n",X3_str[i], X3_tr[i], dX3_str[i]);
	}


	y = soap_kernel(X2_str, X3_str, X2_tr, X3_tr,
			 beta_2, beta_3, xi_3, size_X2, size_X3);

	dy = der_soap_kernel(dX2_str, dX3_str, X2_str, X3_str, X2_tr, X3_tr,
			  beta_2, beta_3, xi_3, size_X2, size_X3);

	double xx[size_X2];

	for (int i = 0; i < size_X2; i++){
		xx[i] = i;
		printf("%f\n",xx[i]);
	}

	printf("y: %f, dy: %f\n",y, dy);

	free(X2_str);
	free(X2_tr);
	free(dX2_str);
	free(X3_str);
	free(X3_tr);
	free(dX3_str);
	return 0;
}