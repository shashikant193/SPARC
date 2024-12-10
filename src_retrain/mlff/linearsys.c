#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
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
#include "sparsification.h"
#include "regression.h"
#include "isddft.h"
#include "ddbp_tools.h"
#include "linearsys.h"
#include "sparc_mlff_interface.h"

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))
//#define au2GPa 29421.02648438959

/*
soap_kernel function computes the SOAP kernel between two descriptors {X2_str, X3_str} and {X2_tr, X3_tr}

[Input]
1. X2_str: pointer to the first descriptor (2-body)
2. X3_str: pointer to the first descriptor (3-body)
3. X2_tr: pointer to the second descriptor (2-body)
4. X3_tr: pointer to the second descriptor (3-body)
5. beta_2: weight to the 2-body term in the kernel
6. beta_3: weight to the 3-body term in the kernel
7. xi_3: exponent in the kernel
8. size_X2: length of the 2-body kernel
9. size_X3: length of the 3-body kernel
[Output]
1. kernel_val: Value of the kernel
*/


double soap_kernel(int kernel_typ, double *X2_str, double *X3_str, double *X2_tr, double *X3_tr,
			 double beta_2, double beta_3, double xi_3, int size_X2, int size_X3) {
	double kernel_val;
	if (kernel_typ ==0){// polynomial kernel implemented in VASP MLFF scheme
		kernel_val = soap_kernel_polynomial(X2_str, X3_str, X2_tr, X3_tr, beta_2, beta_3, xi_3, size_X2, size_X3);
	} else if (kernel_typ==1){//  Gaussiam Kernel
		kernel_val = soap_kernel_Gaussian(X2_str, X3_str, X2_tr, X3_tr, beta_2, beta_3, xi_3, size_X2, size_X3);
	} else{// Laplacian Kernel	
		kernel_val = soap_kernel_Laplacian(X2_str, X3_str, X2_tr, X3_tr, beta_2, beta_3, xi_3, size_X2, size_X3);
	} 	
	if (kernel_val>1.0+1e-5){
		printf("Error in soap kernel evaluation(>1) error %f\n", kernel_val);
		exit(1);
	}
	return kernel_val; 

}

double der_soap_kernel(int kernel_typ, double *dX2_str, double *dX3_str, double *X2_str, double *X3_str, double *X2_tr, double *X3_tr,
			 double beta_2, double beta_3, double xi_3, int size_X2, int size_X3) {
	// printf("kernel_typ: %d, der_soap_kernel_polynomial: %f\n",kernel_typ,der_soap_kernel_polynomial(dX2_str, dX3_str, X2_str, X3_str, X2_tr, X3_tr, beta_2, beta_3, xi_3, size_X2, size_X3));
	// exit(1);
	if (kernel_typ ==0)
		return der_soap_kernel_polynomial(dX2_str, dX3_str, X2_str, X3_str, X2_tr, X3_tr, beta_2, beta_3, xi_3, size_X2, size_X3);
	else if (kernel_typ==1)
		return der_soap_kernel_Gaussian(dX2_str, dX3_str, X2_str, X3_str, X2_tr, X3_tr, beta_2, beta_3, xi_3, size_X2, size_X3);
	else
		return der_soap_kernel_Laplacian(dX2_str, dX3_str, X2_str, X3_str, X2_tr, X3_tr, beta_2, beta_3, xi_3, size_X2, size_X3);
}

double soap_kernel_polynomial(double *X2_str, double *X3_str, double *X2_tr, double *X3_tr,
			 double beta_2, double beta_3, double xi_3, int size_X2, int size_X3) {

	double norm_X3_str, norm_X3_tr, X3_str_temp[size_X3], X3_tr_temp[size_X3], kernel_val;

	norm_X3_str = sqrt(dotProduct(X3_str, X3_str, size_X3));
	norm_X3_tr = sqrt(dotProduct(X3_tr, X3_tr, size_X3));
	for (int i = 0; i<size_X3; i++){
		X3_str_temp[i] = X3_str[i]/norm_X3_str;
		X3_tr_temp[i] = X3_tr[i]/norm_X3_tr;
	}

	kernel_val = beta_2 * dotProduct(X2_str, X2_tr, size_X2) + beta_3 * pow(dotProduct(X3_tr_temp, X3_str_temp, size_X3), xi_3);
	return kernel_val;
}

double soap_kernel_Gaussian(double *X2_str, double *X3_str, double *X2_tr, double *X3_tr,
			 double beta_2, double beta_3, double xi_3, int size_X2, int size_X3) {

	double norm_X3_str, norm_X3_tr, X3_str_temp[size_X3], X3_tr_temp[size_X3], kernel_val;
	int i;
	norm_X3_str = sqrt(dotProduct(X3_str, X3_str, size_X3));
	norm_X3_tr = sqrt(dotProduct(X3_tr, X3_tr, size_X3));
	for (i = 0; i<size_X3; i++){
		X3_str_temp[i] = X3_str[i]/norm_X3_str;
		X3_tr_temp[i] = X3_tr[i]/norm_X3_tr;
	}
	double er_X2[size_X2], er_X3[size_X3];
	for (i=0; i <size_X2; i++)
		er_X2[i] = X2_str[i]-X2_tr[i];
	for (i=0; i <size_X3; i++)
		er_X3[i] = X3_str_temp[i]-X3_tr_temp[i];
	kernel_val = beta_2 * exp(-0.5*dotProduct(er_X2, er_X2, size_X2)) + beta_3 * exp(-0.5*dotProduct(er_X3, er_X3, size_X3));
	// printf("Kernel Gausian: %f\n",kernel_val);
	// kernel_val = beta_2 * dotProduct(X2_str, X2_tr, size_X2) + beta_3 * pow(dotProduct(X3_tr_temp, X3_str_temp, size_X3), xi_3);
	return kernel_val;
}

double soap_kernel_Laplacian(double *X2_str, double *X3_str, double *X2_tr, double *X3_tr,
			 double beta_2, double beta_3, double xi_3, int size_X2, int size_X3) {

	double norm_X3_str, norm_X3_tr, X3_str_temp[size_X3], X3_tr_temp[size_X3], kernel_val;
	int i;

	double er_X2[size_X2], er_X3[size_X3];
	for (i=0; i <size_X2; i++)
		er_X2[i] = X2_str[i]-X2_tr[i];
	for (i=0; i <size_X3; i++)
		er_X3[i] = X3_str[i]-X3_tr[i];
	kernel_val = beta_2 * exp(-0.5*sqrt(dotProduct(er_X2, er_X2, size_X2))) + beta_3 * exp(-0.5*sqrt(dotProduct(er_X3, er_X3, size_X3)));
	// kernel_val = beta_2 * dotProduct(X2_str, X2_tr, size_X2) + beta_3 * pow(dotProduct(X3_tr_temp, X3_str_temp, size_X3), xi_3);
	return kernel_val;
}

/*
der_soap_kernel function computes the derivative of the kernel w.r.t to some variable

[Input]
1. dX2_str: pointer to the derivative of first descriptor w.r.t to the given variable (2-body)
2. dX3_str: pointer to the derivative of first descriptor w.r.t to the given variable (3-body)
3. X2_str: pointer to the first descriptor (2-body)
4. X3_str: pointer to the first descriptor (3-body)
5. X2_tr: pointer to the second descriptor (2-body)
6. X3_tr: pointer to the second descriptor (3-body)
7. beta_2: weight to the 2-body term in the kernel
8. beta_3: weight to the 3-body term in the kernel
9. xi_3: exponent in the kernel
10. size_X2: length of the 2-body kernel
11. size_X3: length of the 3-body kernel
[Output]
1. der_val: derivative of the kernel
*/

double der_soap_kernel_polynomial(double *dX2_str, double *dX3_str, double *X2_str, double *X3_str, double *X2_tr, double *X3_tr,
			 double beta_2, double beta_3, double xi_3, int size_X2, int size_X3) {

	double norm_X3_str, norm_X3_tr, const1, der_val, const2, X3_str_temp[size_X3], X3_tr_temp[size_X3], temp, temp1, temp0;

	norm_X3_str = sqrt(dotProduct(X3_str, X3_str, size_X3));
	norm_X3_tr = sqrt(dotProduct(X3_tr, X3_tr, size_X3));

	for (int i = 0; i<size_X3; i++){
		X3_str_temp[i] = X3_str[i]/norm_X3_str;
		X3_tr_temp[i] = X3_tr[i]/norm_X3_tr;
	}

	temp = pow(norm_X3_str, -1-xi_3);

	temp0 = dotProduct(X3_str, X3_tr_temp, size_X3);

	temp1 = pow(temp0, xi_3-1);

	const1 = -1.0 * beta_3* xi_3 * temp * temp1*temp0;
	const2 = beta_3 * temp * norm_X3_str * xi_3 * temp1;

	der_val = beta_2 * dotProduct(X2_tr, dX2_str, size_X2) + 
			const1*dotProduct(X3_str_temp, dX3_str, size_X3) + const2*dotProduct(X3_tr_temp, dX3_str, size_X3);
	
	return der_val;
}

double der_soap_kernel_Gaussian(double *dX2_str, double *dX3_str, double *X2_str, double *X3_str, double *X2_tr, double *X3_tr,
			 double beta_2, double beta_3, double xi_3, int size_X2, int size_X3) {

	double norm_X3_str, norm_X3_tr, const1, der_val, const2, X3_str_temp[size_X3], X3_tr_temp[size_X3], temp, temp1, temp0;

	double temp2[size_X2], temp3[size_X3];

	double er_X2[size_X2], er_X3[size_X3];

	norm_X3_str = sqrt(dotProduct(X3_str, X3_str, size_X3));
	norm_X3_tr = sqrt(dotProduct(X3_tr, X3_tr, size_X3));

	double norm_X3_str2 = norm_X3_str*norm_X3_str, norm_X3_str3 = norm_X3_str*norm_X3_str*norm_X3_str;

	for (int i = 0; i<size_X3; i++){
		X3_str_temp[i] = X3_str[i]/norm_X3_str;
		X3_tr_temp[i] = X3_tr[i]/norm_X3_tr;
	}

	for (int i=0; i <size_X2; i++)
		er_X2[i] = X2_str[i]-X2_tr[i];
	for (int i=0; i <size_X3; i++)
		er_X3[i] = X3_str_temp[i]-X3_tr_temp[i];

	double exp_temp3 = exp(-0.5*dotProduct(er_X3, er_X3, size_X3));
	double exp_temp2 = exp(-0.5*dotProduct(er_X2, er_X2, size_X2));
	temp = dotProduct(X3_str_temp,X3_tr_temp,size_X3);

	for (int i=0; i <size_X2; i++)
		temp2[i] = -(X2_str[i]-X2_tr[i])*exp_temp2;
	for (int i=0; i <size_X3; i++){
		temp3[i] = -exp_temp3*(- X3_tr_temp[i] + X3_str_temp[i]*temp)*(1.0/norm_X3_str);
	}
		// temp3[i] = -(X3_str[i]-X3_tr[i])*exp_temp3;

	der_val = beta_2 * dotProduct(temp2, dX2_str, size_X2) + beta_3 * dotProduct(temp3, dX3_str, size_X3);


	// temp = pow(norm_X3_str, -1-xi_3);

	// temp0 = dotProduct(X3_str, X3_tr_temp, size_X3);

	// temp1 = pow(temp0, xi_3-1);

	// const1 = -1 * beta_3* xi_3 * temp * temp1*temp0;
	// const2 = beta_3 * temp * norm_X3_str * xi_3 * temp1;

	// der_val = beta_2 * dotProduct(X2_tr, dX2_str, size_X2) + 
	// 		const1*dotProduct(X3_str_temp, dX3_str, size_X3) + const2*dotProduct(X3_tr_temp, dX3_str, size_X3);
	
	return der_val;
}



double der_soap_kernel_Laplacian(double *dX2_str, double *dX3_str, double *X2_str, double *X3_str, double *X2_tr, double *X3_tr,
			 double beta_2, double beta_3, double xi_3, int size_X2, int size_X3) {

	double norm_X3_str, norm_X3_tr, const1, der_val, const2, X3_str_temp[size_X3], X3_tr_temp[size_X3], temp, temp1, temp0;


	double temp2[size_X2], temp3[size_X3];
	double er_X2[size_X2], er_X3[size_X3];
	for (int i=0; i <size_X2; i++)
		er_X2[i] = X2_str[i]-X2_tr[i];
	for (int i=0; i <size_X3; i++)
		er_X3[i] = X3_str[i]-X3_tr[i];

	double exp_temp2 = exp(-0.5*sqrt(dotProduct(er_X2, er_X2, size_X2)));
	double exp_temp3 = exp(-0.5*sqrt(dotProduct(er_X3, er_X3, size_X3)));

	for (int i=0; i <size_X2; i++){
		if ((X2_str[i]-X2_tr[i])>0)
			temp = -0.5;
		else if ((X2_str[i]-X2_tr[i])<0)
			temp = 0.5;
		else
			temp = 0.0;
		temp2[i] = temp*exp_temp2;
	}
	for (int i=0; i <size_X3; i++){
		if ((X3_str[i]-X3_tr[i])>0)
			temp = -0.5;
		else if ((X3_str[i]-X3_tr[i])<0)
			temp = 0.5;
		else
			temp = 0.0;
		temp3[i] = temp*exp_temp3;
	}

	der_val = beta_2 * dotProduct(temp2, dX2_str, size_X2) + beta_3 * dotProduct(temp3, dX3_str, size_X3);



	// temp = pow(norm_X3_str, -1-xi_3);

	// temp0 = dotProduct(X3_str, X3_tr_temp, size_X3);

	// temp1 = pow(temp0, xi_3-1);

	// const1 = -1 * beta_3* xi_3 * temp * temp1*temp0;
	// const2 = beta_3 * temp * norm_X3_str * xi_3 * temp1;

	// der_val = beta_2 * dotProduct(X2_tr, dX2_str, size_X2) + 
	// 		const1*dotProduct(X3_str_temp, dX3_str, size_X3) + const2*dotProduct(X3_tr_temp, dX3_str, size_X3);
	
	return der_val;
}




/*
copy_descriptors function copies the content of one SoapObj to another

[Input]
1. soap_str: SoapObj structure to be copied
[Output]
1. soap_str_MLFF: SoapObj structure where it needs to be copied
*/

void copy_descriptors(SoapObj *soap_str_MLFF, SoapObj *soap_str){

	soap_str_MLFF->natom = soap_str->natom;
	soap_str_MLFF->rcut = soap_str->rcut;
	// soap_str_MLFF->cell[0] = soap_str->cell[0];
	// soap_str_MLFF->cell[1] = soap_str->cell[1];
	// soap_str_MLFF->cell[2] = soap_str->cell[2];
	soap_str_MLFF->cell_measure = soap_str->cell_measure;
	soap_str_MLFF->Lmax = soap_str->Lmax;
	soap_str_MLFF->Nmax = soap_str->Nmax;
	soap_str_MLFF->size_X2 = soap_str->size_X2;
	soap_str_MLFF->size_X3 = soap_str->size_X3;
	soap_str_MLFF->beta_2 = soap_str->beta_2;
	soap_str_MLFF->beta_3 = soap_str->beta_3;
	soap_str_MLFF->xi_3 = soap_str->xi_3;
	soap_str_MLFF->nelem = soap_str->nelem;
	soap_str_MLFF->N_rgrid = soap_str->N_rgrid;
	soap_str_MLFF->natom_domain = soap_str->natom_domain;
	//soap_str_MLFF->atom_idx_domain = (int *) malloc(sizeof(int)*soap_str->natom_domain);
	//soap_str_MLFF->el_idx_domain = (int *) malloc(sizeof(int)*soap_str->natom_domain);
	for (int i=0; i < soap_str->natom_domain; i++){
		soap_str_MLFF->atom_idx_domain[i] = soap_str->atom_idx_domain[i];
		soap_str_MLFF->el_idx_domain[i] = soap_str->el_idx_domain[i];
	}

	for (int i = 0; i < soap_str->natom_domain; i++){
		soap_str_MLFF->Nneighbors[i] = soap_str->Nneighbors[i];
	}

	for (int i = 0; i < soap_str->natom_domain; i++){
		soap_str_MLFF->unique_Nneighbors[i] = soap_str->unique_Nneighbors[i];
	}

	for (int i = 0; i < soap_str->nelem; i++){
		soap_str_MLFF->natom_elem[i] = soap_str->natom_elem[i];
	}

	for (int i = 0; i < soap_str->natom_domain; i++){
		for (int j = 0; j < soap_str->nelem; j++){
			soap_str_MLFF->unique_Nneighbors_elemWise[i][j] = soap_str->unique_Nneighbors_elemWise[i][j];
		}
	}

	for (int i = 0; i < soap_str->natom_domain; i++){
		soap_str_MLFF->neighborList[i].len = soap_str->neighborList[i].len;
		//soap_str_MLFF->neighborList[i].array = (int*)malloc(sizeof(int)*soap_str->neighborList[i].len);
		soap_str_MLFF->unique_neighborList[i].len = soap_str->unique_neighborList[i].len;
		//soap_str_MLFF->unique_neighborList[i].array = (int*)malloc(sizeof(int)*soap_str->unique_neighborList[i].len);
		for (int l = 0; l < soap_str->neighborList[i].len; l++){
			soap_str_MLFF->neighborList[i].array[l] = soap_str->neighborList[i].array[l];
		}
		for (int l = 0; l < soap_str->unique_neighborList[i].len; l++){
			soap_str_MLFF->unique_neighborList[i].array[l] = soap_str->unique_neighborList[i].array[l];
		}
		for (int j = 0; j < soap_str->nelem; j++){
			soap_str_MLFF->unique_neighborList_elemWise[i][j].len = soap_str->unique_neighborList_elemWise[i][j].len;
			//soap_str_MLFF->unique_neighborList_elemWise[i][j].array = (int*)malloc(sizeof(int)*soap_str->unique_neighborList_elemWise[i][j].len);
			for (int k = 0; k < soap_str->unique_neighborList_elemWise[i][j].len; k++){
				soap_str_MLFF->unique_neighborList_elemWise[i][j].array[k] = soap_str->unique_neighborList_elemWise[i][j].array[k];
			}
		}
	}


	for (int i = 0; i < soap_str->natom_domain; i++){
		for (int j = 0; j < soap_str->size_X2; j++){
			soap_str_MLFF->X2[i][j] = soap_str->X2[i][j];
		}
		for (int j = 0; j < soap_str->size_X3; j++){
			soap_str_MLFF->X3[i][j] = soap_str->X3[i][j];
		}
		int uniq_natms = uniqueEle((soap_str->neighborList[i]).array, soap_str->Nneighbors[i]);
		for (int j = 0; j < 1+uniq_natms; j++){
			for (int k = 0; k < soap_str->size_X2; k++){
				soap_str_MLFF->dX2_dX[i][j][k] = soap_str->dX2_dX[i][j][k];
				soap_str_MLFF->dX2_dY[i][j][k] = soap_str->dX2_dY[i][j][k];
				soap_str_MLFF->dX2_dZ[i][j][k] = soap_str->dX2_dZ[i][j][k];
			}
			for (int k = 0; k < soap_str->size_X3; k++){
				soap_str_MLFF->dX3_dX[i][j][k] = soap_str->dX3_dX[i][j][k];
				soap_str_MLFF->dX3_dY[i][j][k] = soap_str->dX3_dY[i][j][k];
				soap_str_MLFF->dX3_dZ[i][j][k] = soap_str->dX3_dZ[i][j][k];
			}
		}
		for (int j = 0; j < 6; j++){
			for (int k = 0; k < soap_str->size_X2; k++){
				soap_str_MLFF->dX2_dF[i][j][k] = soap_str->dX2_dF[i][j][k];
			}
			for (int k = 0; k < soap_str->size_X3; k++){
				soap_str_MLFF->dX3_dF[i][j][k] = soap_str->dX3_dF[i][j][k];
			}
		}

	}
}

/*
add_firstMD function updates the MLFF_Obj by updating design matrix, b vector etc. for the first MD

[Input]
1. soap_str: SoapObj structure of the first MD
2. nlist: NeighList strcuture of the first MD
3. mlff_str: MLFF_Obj structure
4. E: energy per atom of first MD structure (Ha/atom)
5. F: atomic foces of first MD structure (Ha/bohr) [ColMajor]
6. stress: stress of first MD structure (GPa)
[Output]
1. mlff_str: MLFF_Obj structure
*/


void add_firstMD(SoapObj *soap_str, NeighList *nlist, MLFF_Obj *mlff_str, double E, double *F, double *stress_sparc) {

	int rank, nprocs;
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	
	int size_X2 = soap_str->size_X2;
	int size_X3 = soap_str->size_X3;
	int natom = soap_str->natom;
	int nelem = soap_str->nelem;
	
	double beta_2 = soap_str->beta_2;
	double beta_3 = soap_str->beta_3;
	double xi_3 = soap_str->xi_3;

	int row_idx, col_idx, count, atom_idx, idx;
	int iatm_str, istress;

	

	// double *stress;
	// stress = (double *)malloc(6*sizeof(double));
	// stress[0] = stress1[0];
	// stress[1] = stress1[3];
	// stress[2] = stress1[5];
	// stress[3] = stress1[1];
	// stress[4] = stress1[4];
	// stress[5] = stress1[2];


	// calculate mean and std deviation to do the normalization
	mlff_str->E_store[mlff_str->E_store_counter] = E;
	for (int i = 0; i < mlff_str->stress_len; i++)
		mlff_str->stress_store[i][mlff_str->E_store_counter] = stress_sparc[i];
	// mlff_str->stress0_store[mlff_str->E_store_counter] = stress_sparc[0];
	// mlff_str->stress1_store[mlff_str->E_store_counter] = stress_sparc[1];
	// mlff_str->stress2_store[mlff_str->E_store_counter] = stress_sparc[2];
	// mlff_str->stress3_store[mlff_str->E_store_counter] = stress_sparc[3];
	// mlff_str->stress4_store[mlff_str->E_store_counter] = stress_sparc[4];
	// mlff_str->stress5_store[mlff_str->E_store_counter] = stress_sparc[5];

	for (int i=0; i < 3*natom; i++){
		mlff_str->F_store[mlff_str->F_store_counter+i] = F[i];
	}
	mlff_str->E_store_counter += 1;
	mlff_str->F_store_counter += 3*natom;

	int kernel_typ = mlff_str->kernel_typ;
	mlff_str->mu_E  = E;
	mlff_str->std_E  = 1;
	mlff_str->std_F  = sqrt(get_variance(F, 3*natom));
	for (int i = 0; i < mlff_str->stress_len; i++){
		mlff_str->std_stress[i]  = 1;
	}

	// populate bvec
	if (rank==0){
		mlff_str->b_no_norm[0] = E;
		for (int istress=0; istress < mlff_str->stress_len; istress++){
			mlff_str->b_no_norm[1+istress] =  stress_sparc[istress];
		}

		for (int i = 0; i < soap_str->natom_domain; i++){
			idx = soap_str->atom_idx_domain[i];
			mlff_str->b_no_norm[3*i+1+mlff_str->stress_len] = F[3*idx];
			mlff_str->b_no_norm[3*i+2+mlff_str->stress_len] = F[3*idx+1];
			mlff_str->b_no_norm[3*i+3+mlff_str->stress_len] = F[3*idx+2];

		}
	} else{
		for (int i = 0; i < soap_str->natom_domain; i++){
			idx = soap_str->atom_idx_domain[i];
			mlff_str->b_no_norm[3*i] = F[3*idx];
			mlff_str->b_no_norm[3*i+1] = F[3*idx+1];
			mlff_str->b_no_norm[3*i+2] = F[3*idx+2];
		}
	}

	if (rank==0){
		mlff_str->variable_type_identifier[mlff_str->n_rows] = 0;

		for (int i = 0; i < mlff_str->stress_len; i++){
			mlff_str->variable_type_identifier[mlff_str->n_rows+1+i] = 2+i;
		}

		for (int i = 0; i < 3*soap_str->natom_domain; i++){
			mlff_str->variable_type_identifier[mlff_str->n_rows+1+ mlff_str->stress_len+ i] = 1;
		}
		
	}

	
	int *cum_natm_elem = (int *)malloc(sizeof(int)*nelem);
	
	cum_natm_elem[0] = 0;
	for (int i = 1; i < nelem; i++){
		cum_natm_elem[i] = cum_natm_elem[i-1]+soap_str->natom_elem[i-1];
	}

	double *X2_gathered, *X3_gathered;

	X2_gathered = (double *) malloc(sizeof(double)*size_X2*soap_str->natom);
	X3_gathered = (double *) malloc(sizeof(double)*size_X3*soap_str->natom);

	double *X2_local, *X3_local;

	X2_local = (double *) malloc(sizeof(double)*size_X2*soap_str->natom_domain);
	X3_local = (double *) malloc(sizeof(double)*size_X3*soap_str->natom_domain);

	for (int i=0; i < soap_str->natom_domain; i++){
		for (int j=0; j < size_X2; j++){
			X2_local[i*size_X2+j] = soap_str->X2[i][j];
		}
		for (int j=0; j < size_X3; j++){
			X3_local[i*size_X3+j] = soap_str->X3[i][j];
		}
	}

	int local_natoms[nprocs];
	MPI_Allgather(&soap_str->natom_domain, 1, MPI_INT, local_natoms, 1, MPI_INT, MPI_COMM_WORLD);

	int recvcounts_X2[nprocs], recvcounts_X3[nprocs], displs_X2[nprocs], displs_X3[nprocs];
	displs_X2[0] = 0;
	displs_X3[0] = 0;
	for (int i=0; i < nprocs; i++){
		recvcounts_X2[i] = local_natoms[i]*size_X2;
		recvcounts_X3[i] = local_natoms[i]*size_X3;
		if (i>0){
			displs_X2[i] = displs_X2[i-1]+local_natoms[i-1]*size_X2;
			displs_X3[i] = displs_X3[i-1]+local_natoms[i-1]*size_X3;
		}
	}


	MPI_Allgatherv(X2_local, size_X2*soap_str->natom_domain, MPI_DOUBLE, X2_gathered, recvcounts_X2, displs_X2, MPI_DOUBLE, MPI_COMM_WORLD);
	MPI_Allgatherv(X3_local, size_X3*soap_str->natom_domain, MPI_DOUBLE, X3_gathered, recvcounts_X3, displs_X3, MPI_DOUBLE, MPI_COMM_WORLD);

	double **X2_gathered_2D = (double **) malloc(sizeof(double*)*natom);
	double **X3_gathered_2D = (double **) malloc(sizeof(double*)*natom);

	for (int i=0; i < natom; i++){
		X2_gathered_2D[i] = (double *) malloc(sizeof(double)*size_X2);
		X3_gathered_2D[i] = (double *) malloc(sizeof(double)*size_X3);
		for (int j=0; j < size_X2; j++){
			X2_gathered_2D[i][j] = X2_gathered[i*size_X2+j];
		}
		for (int j=0; j < size_X3; j++){
			X3_gathered_2D[i][j] = X3_gathered[i*size_X3+j];
		}
	}

	dyArray *highrank_ID_descriptors = (dyArray *) malloc(sizeof(dyArray)*nelem);
	
	for (int i=0; i <nelem; i++){
		int N_low_min = soap_str->natom_elem[i] - 500;
		init_dyarray(&highrank_ID_descriptors[i]);
		SOAP_CUR_sparsify(kernel_typ, &X2_gathered_2D[cum_natm_elem[i]], &X3_gathered_2D[cum_natm_elem[i]],
				 soap_str->natom_elem[i], size_X2, size_X3, beta_2, beta_3, xi_3, &highrank_ID_descriptors[i], N_low_min);
	}
	
	for (int i = 0; i < nelem; i++){
		mlff_str->natm_train_elemwise[i] = (highrank_ID_descriptors[i]).len;
	}

	count=0;
	for (int i = 0; i < nelem; i++){
		for (int j = 0; j < (highrank_ID_descriptors[i]).len; j++){
			mlff_str->natm_typ_train[count] = i;
			for(int jj = 0; jj < size_X2; jj++){
				mlff_str->X2_traindataset[count][jj] = X2_gathered_2D[cum_natm_elem[i]+(highrank_ID_descriptors[i]).array[j]][jj];
			}
			for(int jj = 0; jj < size_X3; jj++){
				mlff_str->X3_traindataset[count][jj] = X3_gathered_2D[cum_natm_elem[i]+(highrank_ID_descriptors[i]).array[j]][jj];
			}
			count++;
		}
	}

	initialize_soapObj(mlff_str->soap_descriptor_strdataset, nlist, soap_str->Lmax, soap_str->Nmax, mlff_str->N_rgrid, soap_str->beta_3, soap_str->xi_3);
	copy_descriptors(mlff_str->soap_descriptor_strdataset, soap_str);


	mlff_str->n_str = 1;
	mlff_str->natm_train_total = count;

	if (rank==0){
		mlff_str->n_rows = 3*soap_str->natom_domain + 1 + mlff_str->stress_len;
	} else{
		mlff_str->n_rows = 3*soap_str->natom_domain;
	}
	

	mlff_str->n_cols = count;


	int *cum_natm_elem1 = (int *)malloc(sizeof(int)*nelem);
	cum_natm_elem1[0] = 0;
	for (int i = 1; i < nelem; i++){
		cum_natm_elem1[i] = cum_natm_elem1[i-1] + mlff_str->natm_train_elemwise[i-1];
	}


	double *K_train_local = (double *) malloc(sizeof(double)*mlff_str->n_cols*(3*natom+1+mlff_str->stress_len)); // row major;
	for (int i=0; i < mlff_str->n_cols*(3*natom+1+mlff_str->stress_len); i++){
		K_train_local[i] = 0.0;
	}

	// Energy term to populate K_train matrix
	int el_type;
	double temp = (1.0/ (double) soap_str->natom);
	for (int i=0; i < soap_str->natom_domain; i++){
		el_type = soap_str->el_idx_domain[i];
		for (int iatm_el_tr = 0; iatm_el_tr < (highrank_ID_descriptors[el_type]).len; iatm_el_tr++){
			col_idx = cum_natm_elem1[el_type] + iatm_el_tr;
			K_train_local[col_idx] += temp * soap_kernel(kernel_typ, soap_str->X2[i], soap_str->X3[i],
														 mlff_str->X2_traindataset[col_idx], mlff_str->X3_traindataset[col_idx],
												 		 beta_2, beta_3, xi_3, size_X2, size_X3);
		}
	}

	// Force term to populate K_train matrix
	int atm_idx;
	for (int i=0; i <soap_str->natom_domain; i++){
		el_type = soap_str->el_idx_domain[i];
		atm_idx = soap_str->atom_idx_domain[i];
		for (int iatm_el_tr = 0; iatm_el_tr < (highrank_ID_descriptors[el_type]).len; iatm_el_tr++){
			col_idx = cum_natm_elem1[el_type] + iatm_el_tr;
			row_idx = 3*atm_idx+1;

			K_train_local[row_idx*mlff_str->n_cols + col_idx] +=  
				der_soap_kernel(kernel_typ, soap_str->dX2_dX[i][0], soap_str->dX3_dX[i][0],
				soap_str->X2[i], soap_str->X3[i],
				mlff_str->X2_traindataset[col_idx], mlff_str->X3_traindataset[col_idx],
				beta_2, beta_3, xi_3, size_X2, size_X3);

			K_train_local[(row_idx+1)*mlff_str->n_cols + col_idx] +=  
				der_soap_kernel(kernel_typ, soap_str->dX2_dY[i][0], soap_str->dX3_dY[i][0],
				soap_str->X2[i], soap_str->X3[i],
				mlff_str->X2_traindataset[col_idx], mlff_str->X3_traindataset[col_idx],
				beta_2, beta_3, xi_3, size_X2, size_X3);

			K_train_local[(row_idx+2)*mlff_str->n_cols + col_idx] +=  
				der_soap_kernel(kernel_typ, soap_str->dX2_dZ[i][0], soap_str->dX3_dZ[i][0],
				soap_str->X2[i], soap_str->X3[i],
				mlff_str->X2_traindataset[col_idx], mlff_str->X3_traindataset[col_idx],
				beta_2, beta_3, xi_3, size_X2, size_X3);

			for (int neighs =0; neighs < soap_str->unique_Nneighbors[i]; neighs++){
				row_idx = 3*soap_str->unique_neighborList[i].array[neighs]+1;
				K_train_local[row_idx*mlff_str->n_cols + col_idx] +=  
					der_soap_kernel(kernel_typ, soap_str->dX2_dX[i][1+neighs], soap_str->dX3_dX[i][1+neighs],
					soap_str->X2[i], soap_str->X3[i],
					mlff_str->X2_traindataset[col_idx], mlff_str->X3_traindataset[col_idx],
					beta_2, beta_3, xi_3, size_X2, size_X3);

				K_train_local[(row_idx+1)*mlff_str->n_cols + col_idx] +=  
					der_soap_kernel(kernel_typ, soap_str->dX2_dY[i][1+neighs], soap_str->dX3_dY[i][1+neighs],
					soap_str->X2[i], soap_str->X3[i],
					mlff_str->X2_traindataset[col_idx], mlff_str->X3_traindataset[col_idx],
					beta_2, beta_3, xi_3, size_X2, size_X3);

				K_train_local[(row_idx+2)*mlff_str->n_cols + col_idx] +=  
					der_soap_kernel(kernel_typ, soap_str->dX2_dZ[i][1+neighs], soap_str->dX3_dZ[i][1+neighs],
					soap_str->X2[i], soap_str->X3[i],
					mlff_str->X2_traindataset[col_idx], mlff_str->X3_traindataset[col_idx],
					beta_2, beta_3, xi_3, size_X2, size_X3);
			}
			
		}
	}

	double volume = soap_str->cell_measure;

	

	for (int i=0; i < soap_str->natom_domain; i++){
		el_type = soap_str->el_idx_domain[i];
		atm_idx = soap_str->atom_idx_domain[i];
		for (int iatm_el_tr = 0; iatm_el_tr < (highrank_ID_descriptors[el_type]).len; iatm_el_tr++){
			col_idx = cum_natm_elem1[el_type] + iatm_el_tr;
			for (int istress = 0; istress < mlff_str->stress_len; istress++){
				row_idx = 3*soap_str->natom+1+istress;
				K_train_local[(row_idx)*mlff_str->n_cols + col_idx] +=  (1.0/volume)*
						der_soap_kernel(kernel_typ, soap_str->dX2_dF[i][istress], soap_str->dX3_dF[i][istress],
						soap_str->X2[i], soap_str->X3[i],
						mlff_str->X2_traindataset[col_idx], mlff_str->X3_traindataset[col_idx],
						beta_2, beta_3, xi_3, size_X2, size_X3);
			}
		}
	}


	double *K_train_assembled;
	K_train_assembled = (double *)malloc(sizeof(double)*mlff_str->n_cols*(3*natom+1+mlff_str->stress_len));

	MPI_Allreduce(K_train_local, K_train_assembled, mlff_str->n_cols*(3*natom+1+mlff_str->stress_len), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	int r_idx;
	if (rank==0){
		for (int i=0; i < mlff_str->n_cols; i++){
			mlff_str->K_train[0][i] = K_train_assembled[i];
		}
		for (int istress =0; istress < mlff_str->stress_len; istress++){
			r_idx = 3*soap_str->natom+1+istress;
			for (int i=0; i < mlff_str->n_cols; i++){
				mlff_str->K_train[1+istress][i] = K_train_assembled[r_idx*mlff_str->n_cols + i];
			}
		}
	}

	
	
	for (int i = 0; i < soap_str->natom_domain; i++){
		atm_idx = soap_str->atom_idx_domain[i];
		r_idx = 3*atm_idx+1;

		for (int j=0; j < mlff_str->n_cols; j++){
			if (rank==0){
				mlff_str->K_train[3*i+1+mlff_str->stress_len][j] = K_train_assembled[r_idx*mlff_str->n_cols + j];
				mlff_str->K_train[3*i+2+mlff_str->stress_len][j] = K_train_assembled[(1+r_idx)*mlff_str->n_cols + j];
				mlff_str->K_train[3*i+3+mlff_str->stress_len][j] = K_train_assembled[(2+r_idx)*mlff_str->n_cols + j];
			} else{
				mlff_str->K_train[3*i][j] = K_train_assembled[r_idx*mlff_str->n_cols + j];
				mlff_str->K_train[3*i+1][j] = K_train_assembled[(1+r_idx)*mlff_str->n_cols + j];
				mlff_str->K_train[3*i+2][j] = K_train_assembled[(2+r_idx)*mlff_str->n_cols + j];
			}
			
		}
	}

	for (int i=0; i <nelem; i++){  
		delete_dyarray(&highrank_ID_descriptors[i]);
	} 
	free(highrank_ID_descriptors);
	free(cum_natm_elem);
	free(cum_natm_elem1);
	free(X2_local);
	free(X3_local);
	free(X2_gathered);
	free(X3_gathered);
	for (int i=0; i < natom; i++){
		free(X2_gathered_2D[i]);
		free(X3_gathered_2D[i]);
	}
	free(X2_gathered_2D);
	free(X3_gathered_2D); free(K_train_local); free(K_train_assembled);

	
}

/*
add_newstr_rows function updates the MLFF_Obj by updating design matrix, b vector etc. for a new reference structure

[Input]
1. soap_str: SoapObj structure of the reference structure to be added
2. nlist: NeighList strcuture of the first MD
3. mlff_str: MLFF_Obj structure
4. E: energy per atom of the reference structure to be added (Ha/atom)
5. F: atomic foces of the reference structure to be added (Ha/bohr) [ColMajor]
6. stress: stress of the reference structure to be added (GPa)
[Output]
1. mlff_str: MLFF_Obj structure
*/

void add_newstr_rows(SoapObj *soap_str, NeighList *nlist, MLFF_Obj *mlff_str, double E, double *F, double *stress_sparc) {
	int row_idx, col_idx, atom_idx;
	int num_Fterms_newstr, iel, iatm_str;

	int  natom = soap_str->natom;
	int nelem = soap_str->nelem	;
	int size_X2 = soap_str->size_X2;
	int size_X3 = soap_str->size_X3;
	double beta_2 = soap_str->beta_2;
	double beta_3 = soap_str->beta_3;
	double xi_3 = soap_str->xi_3;





	int kernel_typ = mlff_str->kernel_typ;

	int rank, nprocs;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    mlff_str->E_store[mlff_str->E_store_counter] = E;
    for (int i = 0; i < mlff_str->stress_len; i++)
		mlff_str->stress_store[i][mlff_str->E_store_counter] = stress_sparc[i];


	for (int i=0; i < 3*natom; i++){
		mlff_str->F_store[mlff_str->F_store_counter+i] = F[i];
	}
	mlff_str->E_store_counter += 1;
	mlff_str->F_store_counter += 3*natom;

	if (mlff_str->n_str > 1) {
		mlff_str->mu_E  = get_mean(mlff_str->E_store, mlff_str->E_store_counter);
		mlff_str->std_E  = sqrt(get_variance(mlff_str->E_store, mlff_str->E_store_counter));
		mlff_str->std_F  = sqrt(get_variance(mlff_str->F_store, mlff_str->F_store_counter));
		for (int i = 0; i < mlff_str->stress_len; i++){
			mlff_str->std_stress[i] = sqrt(get_variance(mlff_str->stress_store[i], mlff_str->E_store_counter));
		}
	} else {
		mlff_str->mu_E  = mlff_str->E_store[0];
		mlff_str->std_E  = 1.0;
		mlff_str->std_F  = sqrt(get_variance(mlff_str->F_store, mlff_str->F_store_counter));
		for (int i = 0; i < mlff_str->stress_len; i++){
			mlff_str->std_stress[i] = 1.0;
		}
	}	


	if (rank==0){
		mlff_str->variable_type_identifier[mlff_str->n_rows] = 0;
		for (int i = 0; i < mlff_str->stress_len; i++){
			mlff_str->variable_type_identifier[mlff_str->n_rows+1+i] = 2+i;
		}
		for (int i = 0; i < 3*soap_str->natom_domain; i++){
			mlff_str->variable_type_identifier[mlff_str->n_rows+1+ mlff_str->stress_len+ i] = 1;
		}
	}


	



	int idx;
	if (rank==0){
		mlff_str->b_no_norm[mlff_str->n_rows] = E;
		for (int istress=0; istress < mlff_str->stress_len; istress++){
			mlff_str->b_no_norm[mlff_str->n_rows+1+istress] = stress_sparc[istress];
		}
		for (int i = 0; i < soap_str->natom_domain; i++){
			idx = soap_str->atom_idx_domain[i];
			mlff_str->b_no_norm[mlff_str->n_rows+3*i+1+mlff_str->stress_len] = F[3*idx];
			mlff_str->b_no_norm[mlff_str->n_rows+3*i+2+mlff_str->stress_len] = F[3*idx+1];
			mlff_str->b_no_norm[mlff_str->n_rows+3*i+3+mlff_str->stress_len] = F[3*idx+2];
		}
	} else{
		for (int i = 0; i < soap_str->natom_domain; i++){
			idx = soap_str->atom_idx_domain[i];
			mlff_str->b_no_norm[mlff_str->n_rows+3*i] = F[3*idx];
			mlff_str->b_no_norm[mlff_str->n_rows+3*i+1] = F[3*idx+1];
			mlff_str->b_no_norm[mlff_str->n_rows+3*i+2] = F[3*idx+2];
		}
	}
	
	

	initialize_soapObj(mlff_str->soap_descriptor_strdataset + mlff_str->n_str, nlist, soap_str->Lmax, soap_str->Nmax, mlff_str->N_rgrid, soap_str->beta_3, soap_str->xi_3);
	copy_descriptors(mlff_str->soap_descriptor_strdataset+mlff_str->n_str, soap_str);

	int *cum_natm_elem = (int *)malloc(sizeof(int)*nelem);
	cum_natm_elem[0] = 0;
	for (int i = 1; i < nelem; i++){
		cum_natm_elem[i] = cum_natm_elem[i-1] + soap_str->natom_elem[i-1];
	}

	int **cum_natm_ele_cols = (int **)malloc(sizeof(int*)*nelem);
	for (int i = 0; i <nelem; i++){
		cum_natm_ele_cols[i] = (int *)malloc(sizeof(int)*mlff_str->natm_train_elemwise[i]);
	}

	for (int i = 0; i < nelem; i++){
		int count=0;
		for (int j=0; j < mlff_str->natm_train_total; j++){
			if (mlff_str->natm_typ_train[j] == i){
				cum_natm_ele_cols[i][count] = j;
				count++;
			}
		}
	}


	double temp = (1.0/ (double) natom);
	double *K_train_local = (double *) malloc(sizeof(double)*mlff_str->n_cols*(3*natom+1+mlff_str->stress_len)); // row major
	for (int i=0; i < mlff_str->n_cols*(3*natom+1+mlff_str->stress_len); i++){
		K_train_local[i] = 0.0;
	}

	int el_type;
	for (int i=0; i < soap_str->natom_domain; i++){
		el_type = soap_str->el_idx_domain[i];
		for (int iatm_el_tr = 0; iatm_el_tr < mlff_str->natm_train_elemwise[el_type]; iatm_el_tr++){
			col_idx = cum_natm_ele_cols[el_type][iatm_el_tr];
			K_train_local[col_idx] += temp * soap_kernel(kernel_typ, soap_str->X2[i], soap_str->X3[i],
														 mlff_str->X2_traindataset[col_idx], mlff_str->X3_traindataset[col_idx],
												 		 beta_2, beta_3, xi_3, size_X2, size_X3);
		}
	}

	int atm_idx;
	for (int i=0; i <soap_str->natom_domain; i++){
		el_type = soap_str->el_idx_domain[i];
		atm_idx = soap_str->atom_idx_domain[i];
		for (int iatm_el_tr = 0; iatm_el_tr < mlff_str->natm_train_elemwise[el_type]; iatm_el_tr++){
			col_idx = cum_natm_ele_cols[el_type][iatm_el_tr];
			row_idx = 3*atm_idx+1;
			
			K_train_local[row_idx*mlff_str->n_cols + col_idx] +=  
				der_soap_kernel(kernel_typ, soap_str->dX2_dX[i][0], soap_str->dX3_dX[i][0],
				soap_str->X2[i], soap_str->X3[i],
				mlff_str->X2_traindataset[col_idx], mlff_str->X3_traindataset[col_idx],
				beta_2, beta_3, xi_3, size_X2, size_X3);

			K_train_local[(row_idx+1)*mlff_str->n_cols + col_idx] +=  
				der_soap_kernel(kernel_typ, soap_str->dX2_dY[i][0], soap_str->dX3_dY[i][0],
				soap_str->X2[i], soap_str->X3[i],
				mlff_str->X2_traindataset[col_idx], mlff_str->X3_traindataset[col_idx],
				beta_2, beta_3, xi_3, size_X2, size_X3);

			K_train_local[(row_idx+2)*mlff_str->n_cols + col_idx] +=  
				der_soap_kernel(kernel_typ, soap_str->dX2_dZ[i][0], soap_str->dX3_dZ[i][0],
				soap_str->X2[i], soap_str->X3[i],
				mlff_str->X2_traindataset[col_idx], mlff_str->X3_traindataset[col_idx],
				beta_2, beta_3, xi_3, size_X2, size_X3);

			for (int neighs =0; neighs < soap_str->unique_Nneighbors[i]; neighs++){
				row_idx = 3*soap_str->unique_neighborList[i].array[neighs]+1;
				K_train_local[row_idx*mlff_str->n_cols + col_idx] +=  
					der_soap_kernel(kernel_typ, soap_str->dX2_dX[i][1+neighs], soap_str->dX3_dX[i][1+neighs],
					soap_str->X2[i], soap_str->X3[i],
					mlff_str->X2_traindataset[col_idx], mlff_str->X3_traindataset[col_idx],
					beta_2, beta_3, xi_3, size_X2, size_X3);

				K_train_local[(row_idx+1)*mlff_str->n_cols + col_idx] +=  
				der_soap_kernel(kernel_typ, soap_str->dX2_dY[i][1+neighs], soap_str->dX3_dY[i][1+neighs],
				soap_str->X2[i], soap_str->X3[i],
				mlff_str->X2_traindataset[col_idx], mlff_str->X3_traindataset[col_idx],
				beta_2, beta_3, xi_3, size_X2, size_X3);

				K_train_local[(row_idx+2)*mlff_str->n_cols + col_idx] +=  
					der_soap_kernel(kernel_typ, soap_str->dX2_dZ[i][1+neighs], soap_str->dX3_dZ[i][1+neighs],
					soap_str->X2[i], soap_str->X3[i],
					mlff_str->X2_traindataset[col_idx], mlff_str->X3_traindataset[col_idx],
					beta_2, beta_3, xi_3, size_X2, size_X3);
			}
			
		}
	}


	double volume = soap_str->cell_measure;
	
	for (int i=0; i < soap_str->natom_domain; i++){
		el_type = soap_str->el_idx_domain[i];
		atm_idx = soap_str->atom_idx_domain[i];
		for (int iatm_el_tr = 0; iatm_el_tr < mlff_str->natm_train_elemwise[el_type]; iatm_el_tr++){
			col_idx = cum_natm_ele_cols[el_type][iatm_el_tr];
			for (int istress = 0; istress < mlff_str->stress_len; istress++){
					row_idx = 3*soap_str->natom+1+istress;
					K_train_local[(row_idx)*mlff_str->n_cols + col_idx] +=  (1.0/volume)*
						der_soap_kernel(kernel_typ, soap_str->dX2_dF[i][istress], soap_str->dX3_dF[i][istress],
						soap_str->X2[i], soap_str->X3[i],
						mlff_str->X2_traindataset[col_idx], mlff_str->X3_traindataset[col_idx],
						beta_2, beta_3, xi_3, size_X2, size_X3);				

			}
		}
	}


	double *K_train_assembled;
	K_train_assembled = (double *)malloc(sizeof(double)*mlff_str->n_cols*(3*natom+1+mlff_str->stress_len));

	MPI_Allreduce(K_train_local, K_train_assembled, mlff_str->n_cols*(3*natom+1+mlff_str->stress_len), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	int r_idx;

	if (rank==0){
		for (int i=0; i < mlff_str->n_cols; i++){
			mlff_str->K_train[mlff_str->n_rows][i] = K_train_assembled[i];
		}
		for (int istress =0; istress < mlff_str->stress_len; istress++){
			r_idx = 3*soap_str->natom+1+istress;
			for (int i=0; i < mlff_str->n_cols; i++){
				mlff_str->K_train[mlff_str->n_rows+1+istress][i] = K_train_assembled[r_idx*mlff_str->n_cols + i];
			}
		}
	}

	
	for (int i = 0; i < soap_str->natom_domain; i++){
		atm_idx = soap_str->atom_idx_domain[i];
		r_idx = 3*atm_idx+1;

		for (int j=0; j < mlff_str->n_cols; j++){
			if (rank==0){
				mlff_str->K_train[mlff_str->n_rows+3*i+1+mlff_str->stress_len][j] = K_train_assembled[r_idx*mlff_str->n_cols + j];
				mlff_str->K_train[mlff_str->n_rows+3*i+2+mlff_str->stress_len][j] = K_train_assembled[(1+r_idx)*mlff_str->n_cols + j];
				mlff_str->K_train[mlff_str->n_rows+3*i+3+mlff_str->stress_len][j] = K_train_assembled[(2+r_idx)*mlff_str->n_cols + j];
			} else{
				mlff_str->K_train[mlff_str->n_rows+3*i][j] = K_train_assembled[r_idx*mlff_str->n_cols + j];
				mlff_str->K_train[mlff_str->n_rows+3*i+1][j] = K_train_assembled[(1+r_idx)*mlff_str->n_cols + j];
				mlff_str->K_train[mlff_str->n_rows+3*i+2][j] = K_train_assembled[(2+r_idx)*mlff_str->n_cols + j];
			}
			
		}
	}
	free(K_train_local);
	free(K_train_assembled);


	// updating other MLFF parameters such as number of structures, number of training environment, element typ of training env
	mlff_str->n_str += 1;
	if (rank==0){
		mlff_str->n_rows += 3*soap_str->natom_domain + 1 + mlff_str->stress_len;
	} else {
		mlff_str->n_rows += 3*soap_str->natom_domain;
	}
	free(cum_natm_elem);
	for (int i = 0; i <nelem; i++){
		free(cum_natm_ele_cols[i]);
	}
	free(cum_natm_ele_cols);
	// free(stress);
}


/*
calculate_Kpredict function calculate the design matrix for prediction for a new structure

[Input]
1. soap_str: SoapObj structure of the new structure
2. nlist: NeighList strcuture of the new structure
3. mlff_str: MLFF_Obj structure
[Output]
1. K_predict: design prediction matrix
*/

void calculate_Kpredict(SoapObj *soap_str, NeighList *nlist, MLFF_Obj *mlff_str, double **K_predict){

 	int row_idx, col_idx, atom_idx, iel, iatm_str, istress;
 	double E_scale, F_scale, *stress_scale;

 	int natom = soap_str->natom;
 	int nelem = soap_str->nelem;
	int size_X2 = soap_str->size_X2;
	int size_X3 = soap_str->size_X3;
	double beta_2 = soap_str->beta_2;
	double beta_3 = soap_str->beta_3;
	double xi_3 = soap_str->xi_3;
	stress_scale = (double*) malloc(mlff_str->stress_len*sizeof(double));


	int rank, nproc;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

 	int kernel_typ = mlff_str->kernel_typ;

 	E_scale = mlff_str->E_scale;
 	F_scale = mlff_str->F_scale * mlff_str->relative_scale_F;

 	for (int i=0; i < mlff_str->stress_len; i++){
 		stress_scale[i] =  mlff_str->stress_scale[i] * mlff_str->relative_scale_stress[i];

 	}

 	

 	// stress_scale[0] = mlff_str->stress_scale[0] * mlff_str->relative_scale_stress[0];
 	// stress_scale[1] = mlff_str->stress_scale[1] * mlff_str->relative_scale_stress[1];
 	// stress_scale[2] = mlff_str->stress_scale[2] * mlff_str->relative_scale_stress[2];
 	// stress_scale[3] = mlff_str->stress_scale[3] * mlff_str->relative_scale_stress[3];
 	// stress_scale[4] = mlff_str->stress_scale[4] * mlff_str->relative_scale_stress[4];
 	// stress_scale[5] = mlff_str->stress_scale[5] * mlff_str->relative_scale_stress[5];

 	int *cum_natm_elem = (int *)malloc(sizeof(int)*nelem);
	cum_natm_elem[0] = 0;
	for (int i = 1; i < nelem; i++){
		cum_natm_elem[i] = cum_natm_elem[i-1] + soap_str->natom_elem[i-1];
	}

	int **cum_natm_ele_cols = (int **)malloc(sizeof(int*)*nelem);
	for (int i = 0; i <nelem; i++){
		cum_natm_ele_cols[i] = (int *)malloc(sizeof(int)*mlff_str->natm_train_elemwise[i]);
	}

	for (int i = 0; i < nelem; i++){
		int count=0;
		for (int j=0; j < mlff_str->natm_train_total; j++){
			if (mlff_str->natm_typ_train[j] == i){
				cum_natm_ele_cols[i][count] = j;
				count++;
			}
		}
	}

	// printf("mlff_str->natm_train_total: %d\n",mlff_str->natm_train_total);
	// printf("mlff_str->natm_typ_train[j]:\n");
	// for (int i=0; i < mlff_str->natm_train_total; i++){
	// 	printf("%d\n",mlff_str->natm_typ_train[i]);
	// }

	// MPI_Barrier(MPI_COMM_WORLD);
	// exit(8);



	double temp = (1.0/ (double) soap_str->natom);
	double *K_train_local = (double *) malloc(sizeof(double)*mlff_str->n_cols*(3*natom+1+mlff_str->stress_len)); // row major;
	for (int i=0; i < mlff_str->n_cols*(3*natom+1+mlff_str->stress_len); i++){
		K_train_local[i] = 0.0;
	}

	

	int el_type;
	for (int i=0; i <soap_str->natom_domain; i++){
		el_type = soap_str->el_idx_domain[i];
		for (int iatm_el_tr = 0; iatm_el_tr < mlff_str->natm_train_elemwise[el_type]; iatm_el_tr++){
			col_idx = cum_natm_ele_cols[el_type][iatm_el_tr];
			K_train_local[col_idx] += temp * soap_kernel(kernel_typ, soap_str->X2[i], soap_str->X3[i],
									mlff_str->X2_traindataset[col_idx], mlff_str->X3_traindataset[col_idx],
									beta_2, beta_3, xi_3, size_X2, size_X3);
		}
	}

	


	int atm_idx;
	for (int i=0; i <soap_str->natom_domain; i++){
		el_type = soap_str->el_idx_domain[i];
		atm_idx = soap_str->atom_idx_domain[i];
		for (int iatm_el_tr = 0; iatm_el_tr < mlff_str->natm_train_elemwise[el_type]; iatm_el_tr++){
			col_idx = cum_natm_ele_cols[el_type][iatm_el_tr];
			row_idx = 3*atm_idx+1;

			K_train_local[row_idx*mlff_str->n_cols + col_idx] +=  F_scale*
				der_soap_kernel(kernel_typ, soap_str->dX2_dX[i][0], soap_str->dX3_dX[i][0],
				 soap_str->X2[i], soap_str->X3[i],
				mlff_str->X2_traindataset[col_idx], mlff_str->X3_traindataset[col_idx],
				beta_2, beta_3, xi_3, size_X2, size_X3);

			K_train_local[(row_idx+1)*mlff_str->n_cols + col_idx] +=  F_scale*
				der_soap_kernel(kernel_typ, soap_str->dX2_dY[i][0], soap_str->dX3_dY[i][0],
				 soap_str->X2[i], soap_str->X3[i],
				mlff_str->X2_traindataset[col_idx], mlff_str->X3_traindataset[col_idx],
				beta_2, beta_3, xi_3, size_X2, size_X3);

			K_train_local[(row_idx+2)*mlff_str->n_cols + col_idx] +=  F_scale*
				der_soap_kernel(kernel_typ, soap_str->dX2_dZ[i][0], soap_str->dX3_dZ[i][0],
				 soap_str->X2[i], soap_str->X3[i],
				mlff_str->X2_traindataset[col_idx], mlff_str->X3_traindataset[col_idx],
				beta_2, beta_3, xi_3, size_X2, size_X3);

			for (int neighs =0; neighs < soap_str->unique_Nneighbors[i]; neighs++){
				row_idx = 3*soap_str->unique_neighborList[i].array[neighs]+1;
				K_train_local[row_idx*mlff_str->n_cols + col_idx] +=  F_scale*
					der_soap_kernel(kernel_typ, soap_str->dX2_dX[i][1+neighs], soap_str->dX3_dX[i][1+neighs],
					 soap_str->X2[i], soap_str->X3[i],
					mlff_str->X2_traindataset[col_idx], mlff_str->X3_traindataset[col_idx],
					beta_2, beta_3, xi_3, size_X2, size_X3);

				K_train_local[(row_idx+1)*mlff_str->n_cols + col_idx] += F_scale* 
				der_soap_kernel(kernel_typ, soap_str->dX2_dY[i][1+neighs], soap_str->dX3_dY[i][1+neighs],
				 soap_str->X2[i], soap_str->X3[i],
				mlff_str->X2_traindataset[col_idx], mlff_str->X3_traindataset[col_idx],
				beta_2, beta_3, xi_3, size_X2, size_X3);

				K_train_local[(row_idx+2)*mlff_str->n_cols + col_idx] +=  F_scale*
					der_soap_kernel(kernel_typ, soap_str->dX2_dZ[i][1+neighs], soap_str->dX3_dZ[i][1+neighs],
					 soap_str->X2[i], soap_str->X3[i],
					mlff_str->X2_traindataset[col_idx], mlff_str->X3_traindataset[col_idx],
					beta_2, beta_3, xi_3, size_X2, size_X3);
			}
			
		}
	}

	
	double volume = soap_str->cell_measure;

	for (int i=0; i < soap_str->natom_domain; i++){
		el_type = soap_str->el_idx_domain[i];
		atm_idx = soap_str->atom_idx_domain[i];
		for (int iatm_el_tr = 0; iatm_el_tr < mlff_str->natm_train_elemwise[el_type]; iatm_el_tr++){
			col_idx = cum_natm_ele_cols[el_type][iatm_el_tr];
			for (int istress = 0; istress < mlff_str->stress_len; istress++){
				row_idx = 3*soap_str->natom+1+istress;
				K_train_local[(row_idx)*mlff_str->n_cols + col_idx] +=  (1.0/volume)*stress_scale[istress]*
				der_soap_kernel(kernel_typ, soap_str->dX2_dF[i][istress], soap_str->dX3_dF[i][istress],
				soap_str->X2[i], soap_str->X3[i],
				mlff_str->X2_traindataset[col_idx], mlff_str->X3_traindataset[col_idx],
				beta_2, beta_3, xi_3, size_X2, size_X3);				

			}
		}
	}

	


	double *K_train_assembled;
	K_train_assembled = (double *)malloc(sizeof(double)*mlff_str->n_cols*(3*natom+1+mlff_str->stress_len));


	MPI_Allreduce(K_train_local, K_train_assembled, mlff_str->n_cols*(3*natom+1+mlff_str->stress_len), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	int r_idx;
	if (rank==0){
		for (int i=0; i < mlff_str->n_cols; i++){
			K_predict[0][i] = K_train_assembled[i];
		}
		for (int istress =0; istress < mlff_str->stress_len; istress++){
			r_idx = 3*soap_str->natom+1+istress;
			for (int i=0; i < mlff_str->n_cols; i++){
				K_predict[1+istress][i] = K_train_assembled[r_idx*mlff_str->n_cols + i];
			}
		}
	}
	


	for (int i = 0; i < soap_str->natom_domain; i++){
		atm_idx = soap_str->atom_idx_domain[i];
		r_idx = 3*atm_idx+1;

		for (int j=0; j < mlff_str->n_cols; j++){
			if (rank==0){
				K_predict[3*i+1+mlff_str->stress_len][j] = K_train_assembled[r_idx*mlff_str->n_cols + j];
				K_predict[3*i+2+mlff_str->stress_len][j] = K_train_assembled[(1+r_idx)*mlff_str->n_cols + j];
				K_predict[3*i+3+mlff_str->stress_len][j] = K_train_assembled[(2+r_idx)*mlff_str->n_cols + j];
			} else{
				K_predict[3*i][j] = K_train_assembled[r_idx*mlff_str->n_cols + j];
				K_predict[3*i+1][j] = K_train_assembled[(1+r_idx)*mlff_str->n_cols + j];
				K_predict[3*i+2][j] = K_train_assembled[(2+r_idx)*mlff_str->n_cols + j];
			}
			
		}
	}

	

    

    free(K_train_local);
	free(K_train_assembled);



	free(cum_natm_elem);
	for (int i = 0; i <nelem; i++){
		free(cum_natm_ele_cols[i]);
	}
	free(cum_natm_ele_cols);
	free(stress_scale);

 }

/*
add_newtrain_cols function updates the MLFF_Obj by updating design matrix columns etc. for a new local confiugration

[Input]
1. mlff_str: MLFF_Obj structure
2. X2: 2-body ddescriptor of the new local confiugration
3. X3: 3-body ddescriptor of the new local confiugration
4. elem_typ: Element type of the new local confiugration
[Output]
1. mlff_str: MLFF_Obj structure
*/

 void add_newtrain_cols(double *X2, double *X3, int elem_typ, MLFF_Obj *mlff_str){
	int row_idx, col_idx, atom_idx, istress;
	int nelem = mlff_str->nelem;
	int natom = mlff_str->natom;
	int size_X2 = mlff_str->size_X2;
	int size_X3 = mlff_str->size_X3;
	int kernel_typ = mlff_str->kernel_typ;
	double beta_2 = mlff_str->beta_2;
	double beta_3 = mlff_str->beta_3;
	double xi_3 = mlff_str->xi_3;

	int rank, nprocs;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	
    

	

	// copy the X2 and X3 into train history data 

	for(int j = 0; j < size_X2; j++){
		mlff_str->X2_traindataset[mlff_str->n_cols][j] = X2[j];
	}
	for(int j = 0; j < size_X3; j++){
		mlff_str->X3_traindataset[mlff_str->n_cols][j] = X3[j];
	}

	// printf("came here rank: %d size_X2: %d, size-X3: %d\n",rank,size_X2,size_X3);
	// MPI_Barrier(MPI_COMM_WORLD);
	// exit(15);

	

	


	double *k_local;
	int total_rows = 0;
	for (int istr = 0; istr < mlff_str->n_str; istr++){
		total_rows = total_rows + 1 + 3*(mlff_str->soap_descriptor_strdataset+istr)->natom + mlff_str->stress_len;
	}
	k_local = (double *)malloc(sizeof(double)*total_rows);
	for (int i=0; i< total_rows; i++){
		k_local[i] = 0.0;
	}


	

	
	row_idx=0;
	int el_type;
	double temp;
	for (int istr = 0; istr < mlff_str->n_str; istr++){
		temp = 1.0/ ((double) (mlff_str->soap_descriptor_strdataset+istr)->natom);
		for (int i=0; i <(mlff_str->soap_descriptor_strdataset+istr)->natom_domain; i++){
			el_type = (mlff_str->soap_descriptor_strdataset+istr)->el_idx_domain[i];
			if (el_type==elem_typ){
				k_local[row_idx] +=  temp*
					soap_kernel(kernel_typ, (mlff_str->soap_descriptor_strdataset+istr)->X2[i], (mlff_str->soap_descriptor_strdataset+istr)->X3[i],
					X2, X3, beta_2, beta_3, xi_3, size_X2, size_X3);
			}
		}
		row_idx += 1+3*(mlff_str->soap_descriptor_strdataset+istr)->natom+mlff_str->stress_len;
	}


	// Force term to populate K_train matrix
	row_idx = 0;
	int row_index_F;
	for (int istr = 0; istr < mlff_str->n_str; istr++){
		for (int iatm_str = 0; iatm_str < (mlff_str->soap_descriptor_strdataset+istr)->natom_domain; iatm_str++){
			el_type = (mlff_str->soap_descriptor_strdataset+istr)->el_idx_domain[iatm_str];
			if (el_type==elem_typ){
				atom_idx = (mlff_str->soap_descriptor_strdataset+istr)->atom_idx_domain[iatm_str];
				row_index_F = row_idx + 3*atom_idx+1;
				// x-component (w.r.t itself) "because an atom is not considered it's neighbour hence dealt outside the neighs loop"
				k_local[row_index_F] +=  
					der_soap_kernel(kernel_typ, (mlff_str->soap_descriptor_strdataset+istr)->dX2_dX[iatm_str][0], (mlff_str->soap_descriptor_strdataset+istr)->dX3_dX[iatm_str][0],
					 (mlff_str->soap_descriptor_strdataset+istr)->X2[iatm_str], (mlff_str->soap_descriptor_strdataset+istr)->X3[iatm_str],
					X2, X3, beta_2, beta_3, xi_3, size_X2, size_X3);
				// y-component (w.r.t itself) "because an atom is not considered it's neighbour hence dealt outside the neighs loop"
				k_local[row_index_F+1] +=  
					der_soap_kernel(kernel_typ, (mlff_str->soap_descriptor_strdataset+istr)->dX2_dY[iatm_str][0], (mlff_str->soap_descriptor_strdataset+istr)->dX3_dY[iatm_str][0],
					 (mlff_str->soap_descriptor_strdataset+istr)->X2[iatm_str], (mlff_str->soap_descriptor_strdataset+istr)->X3[iatm_str],
					X2, X3, beta_2, beta_3, xi_3, size_X2, size_X3);
				// z-component (w.r.t itself) "because an atom is not considered it's neighbour hence dealt outside the neighs loop"
				k_local[row_index_F+2] +=  
					der_soap_kernel(kernel_typ, (mlff_str->soap_descriptor_strdataset+istr)->dX2_dZ[iatm_str][0], (mlff_str->soap_descriptor_strdataset+istr)->dX3_dZ[iatm_str][0],
					 (mlff_str->soap_descriptor_strdataset+istr)->X2[iatm_str], (mlff_str->soap_descriptor_strdataset+istr)->X3[iatm_str],
					X2, X3, beta_2, beta_3, xi_3, size_X2, size_X3);

				for (int neighs = 0; neighs < (mlff_str->soap_descriptor_strdataset+istr)->unique_Nneighbors[iatm_str]; neighs++){
					row_index_F = row_idx + 3*(mlff_str->soap_descriptor_strdataset+istr)->unique_neighborList[iatm_str].array[neighs]+1;
					k_local[row_index_F] +=  
					der_soap_kernel(kernel_typ, (mlff_str->soap_descriptor_strdataset+istr)->dX2_dX[iatm_str][1+neighs], (mlff_str->soap_descriptor_strdataset+istr)->dX3_dX[iatm_str][1+neighs],
						 (mlff_str->soap_descriptor_strdataset+istr)->X2[iatm_str], (mlff_str->soap_descriptor_strdataset+istr)->X3[iatm_str],
						X2, X3, beta_2, beta_3, xi_3, size_X2, size_X3);
					// y-component (w.r.t neighs neighbour)
					k_local[row_index_F+1] +=  
					der_soap_kernel(kernel_typ, (mlff_str->soap_descriptor_strdataset+istr)->dX2_dY[iatm_str][1+neighs], (mlff_str->soap_descriptor_strdataset+istr)->dX3_dY[iatm_str][1+neighs],
						 (mlff_str->soap_descriptor_strdataset+istr)->X2[iatm_str], (mlff_str->soap_descriptor_strdataset+istr)->X3[iatm_str],
						X2, X3, beta_2, beta_3, xi_3, size_X2, size_X3);
					// z-component (w.r.t neighs neighbour)
					k_local[row_index_F+2] +=  
					der_soap_kernel(kernel_typ, (mlff_str->soap_descriptor_strdataset+istr)->dX2_dZ[iatm_str][1+neighs], (mlff_str->soap_descriptor_strdataset+istr)->dX3_dZ[iatm_str][1+neighs],
						 (mlff_str->soap_descriptor_strdataset+istr)->X2[iatm_str], (mlff_str->soap_descriptor_strdataset+istr)->X3[iatm_str],
						X2, X3, beta_2, beta_3, xi_3, size_X2, size_X3);
				}
			}
					
		}
		row_idx += 1+3*(mlff_str->soap_descriptor_strdataset+istr)->natom+mlff_str->stress_len;
	}






	int row_index_stress;
	double volume;
	row_idx = 0;
	for (int istr = 0; istr < mlff_str->n_str; istr++){
		volume = (mlff_str->soap_descriptor_strdataset+istr)->cell_measure;
		for (int iatm_str = 0; iatm_str < (mlff_str->soap_descriptor_strdataset+istr)->natom_domain; iatm_str++){
			el_type = (mlff_str->soap_descriptor_strdataset+istr)->el_idx_domain[iatm_str];
			if (el_type==elem_typ){
				for (int istress = 0; istress < mlff_str->stress_len; istress++){
					row_index_stress = row_idx + 3*(mlff_str->soap_descriptor_strdataset+istr)->natom + 1 + istress;
					k_local[row_index_stress] +=   (1.0/volume)*
					der_soap_kernel(kernel_typ, (mlff_str->soap_descriptor_strdataset+istr)->dX2_dF[iatm_str][istress], (mlff_str->soap_descriptor_strdataset+istr)->dX3_dF[iatm_str][istress],
						 (mlff_str->soap_descriptor_strdataset+istr)->X2[iatm_str], (mlff_str->soap_descriptor_strdataset+istr)->X3[iatm_str],
						X2, X3, beta_2, beta_3, xi_3, size_X2, size_X3);				

				}
			}
		}
		row_idx += 1+3*(mlff_str->soap_descriptor_strdataset+istr)->natom+mlff_str->stress_len;
	}





	double *K_train_assembled;
	K_train_assembled = (double *)malloc(sizeof(double)*total_rows);

	MPI_Allreduce(k_local, K_train_assembled, total_rows, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	int temp_idx1 = 0, temp_idx2 = 0;
	if (rank==0){
		for (int i=0; i < mlff_str->n_str; i++){
			
			// mlff_str->K_train[i*(3*mlff_str->natom_domain+1+mlff_str->stress_len)][mlff_str->n_cols] = K_train_assembled[i*(3*natom+1+mlff_str->stress_len)];
			mlff_str->K_train[temp_idx1][mlff_str->n_cols] = K_train_assembled[temp_idx2];
			for (int istress =0; istress < mlff_str->stress_len; istress++){
				// mlff_str->K_train[i*(3*mlff_str->natom_domain+1+mlff_str->stress_len)+1+istress][mlff_str->n_cols] = K_train_assembled[i*(3*natom+1+mlff_str->stress_len) + 3*natom+1+istress];
				mlff_str->K_train[temp_idx1+1+istress][mlff_str->n_cols] = K_train_assembled[temp_idx2 + 3*((mlff_str->soap_descriptor_strdataset+i)->natom)+1+istress];

			}
			temp_idx1 = temp_idx1 + 1 + mlff_str->stress_len + 3*((mlff_str->soap_descriptor_strdataset+i)->natom_domain);
			temp_idx2 = temp_idx2 + 1 + mlff_str->stress_len + 3*((mlff_str->soap_descriptor_strdataset+i)->natom);
		}
	}

	int r_idx, temp_idx, atm_idx;
	temp_idx=0;
	temp_idx1 = 0;
	for (int s=0; s < mlff_str->n_str; s++){
		for (int i = 0; i < (mlff_str->soap_descriptor_strdataset+s)->natom_domain; i++){
			atm_idx = (mlff_str->soap_descriptor_strdataset+s)->atom_idx_domain[i];
			r_idx = 3*atm_idx+1;
			if (rank==0){
				// mlff_str->K_train[temp_idx+3*i+1+mlff_str->stress_len][mlff_str->n_cols] = K_train_assembled[s*(3*natom+1+mlff_str->stress_len)+r_idx];
				// mlff_str->K_train[temp_idx+3*i+2+mlff_str->stress_len][mlff_str->n_cols] = K_train_assembled[s*(3*natom+1+mlff_str->stress_len)+r_idx+1];
				// mlff_str->K_train[temp_idx+3*i+3+mlff_str->stress_len][mlff_str->n_cols] = K_train_assembled[s*(3*natom+1+mlff_str->stress_len)+r_idx+2];
				mlff_str->K_train[temp_idx+3*i+1+mlff_str->stress_len][mlff_str->n_cols] = K_train_assembled[temp_idx1+r_idx];
				mlff_str->K_train[temp_idx+3*i+2+mlff_str->stress_len][mlff_str->n_cols] = K_train_assembled[temp_idx1+r_idx+1];
				mlff_str->K_train[temp_idx+3*i+3+mlff_str->stress_len][mlff_str->n_cols] = K_train_assembled[temp_idx1+r_idx+2];
			} else {
				// mlff_str->K_train[temp_idx+3*i][mlff_str->n_cols] = K_train_assembled[s*(3*natom+1+mlff_str->stress_len)+r_idx];
				// mlff_str->K_train[temp_idx+3*i+1][mlff_str->n_cols] = K_train_assembled[s*(3*natom+1+mlff_str->stress_len)+r_idx+1];
				// mlff_str->K_train[temp_idx+3*i+2][mlff_str->n_cols] = K_train_assembled[s*(3*natom+1+mlff_str->stress_len)+r_idx+2];
				mlff_str->K_train[temp_idx+3*i][mlff_str->n_cols] = K_train_assembled[temp_idx1+r_idx];
				mlff_str->K_train[temp_idx+3*i+1][mlff_str->n_cols] = K_train_assembled[temp_idx1+r_idx+1];
				mlff_str->K_train[temp_idx+3*i+2][mlff_str->n_cols] = K_train_assembled[temp_idx1+r_idx+2];
			}
		}
		if (rank==0){
			temp_idx =  temp_idx + 1 + 3*((mlff_str->soap_descriptor_strdataset+s)->natom_domain) +mlff_str->stress_len;
		} else {
			temp_idx =  temp_idx + 3*((mlff_str->soap_descriptor_strdataset+s)->natom_domain);
		}
		temp_idx1 =  temp_idx1 + 1 + mlff_str->stress_len + 3*((mlff_str->soap_descriptor_strdataset+s)->natom);
		
	}

	


	// updating other MLFF parameters such as number of structures, number of training environment, element typ of training env
	mlff_str->natm_typ_train[mlff_str->n_cols] = elem_typ;
	mlff_str->natm_train_total += 1;
	mlff_str->n_cols += 1;
	mlff_str->natm_train_elemwise[elem_typ] += 1;
	free(K_train_assembled); 
	free(k_local);

   

}

/*
remove_str_rows function removes a given reference structure from the training dataset

[Input]
1. mlff_str: MLFF_Obj structure
2. str_ID: ID of the reference structure in training dataset
[Output]
1. mlff_str: MLFF_Obj structure
*/

void remove_str_rows(MLFF_Obj *mlff_str, int str_ID){
	int start_idx = 0, end_idx = 0, i, j, istr, rows_to_delete;

	for (i = 0; i < str_ID; i++){
		start_idx += 3*(mlff_str->soap_descriptor_strdataset+i)->natom + 1;
	}
	end_idx = start_idx + 3*(mlff_str->soap_descriptor_strdataset+str_ID)->natom + 1;

	rows_to_delete = end_idx - start_idx;
	

	for (istr = str_ID + 1; istr < mlff_str->n_str; istr++){
		copy_descriptors(mlff_str->soap_descriptor_strdataset + istr - 1, mlff_str->soap_descriptor_strdataset + istr );
		// mlff_str->soap_descriptor_strdataset[istr] = mlff_str->soap_descriptor_strdataset[istr + 1];
	}
	delete_soapObj(mlff_str->soap_descriptor_strdataset + mlff_str->n_str, mlff_str->natom_domain);
	mlff_str->n_str = mlff_str->n_str - 1;

	for (i = start_idx; i < end_idx; i++) {
		mlff_str->b_no_norm[i] = mlff_str->b_no_norm[i + rows_to_delete];
		for (j = 0; j < mlff_str->n_cols; j++){
			mlff_str->K_train[i][j] = mlff_str->K_train[i + rows_to_delete][j];
		}
	}

	for (i = mlff_str->n_rows - rows_to_delete; i < mlff_str->n_rows; i++) {
		mlff_str->b_no_norm[i] = 0.0;
		for (j = 0; j < mlff_str->n_cols; j++){
			mlff_str->K_train[i][j] = 0.0;
		}
	}


	mlff_str->n_rows = mlff_str->n_rows - rows_to_delete;
}

/*
remove_train_cols function removes a given local confiugration from the training dataset

[Input]
1. mlff_str: MLFF_Obj structure
2. col_ID: ID of the local confiugration in training dataset
[Output]
1. mlff_str: MLFF_Obj structure
*/
void remove_train_cols(MLFF_Obj *mlff_str, int col_ID){
	int i, j;
	for (i = col_ID; i < mlff_str->n_cols-1; i++){
		for (j = 0; j < mlff_str->n_rows; j++){
			mlff_str->K_train[j][i] = mlff_str->K_train[j][i+1];
		}
	}


	for (j = 0; j < mlff_str->n_rows; j++){
		mlff_str->K_train[j][mlff_str->n_cols-1] = 0.0;
	}

	for (i =col_ID; i < mlff_str->n_cols-1; i++){
		for (j=0; j < mlff_str->size_X3; j++){
			mlff_str->X3_traindataset[i][j] = mlff_str->X3_traindataset[i+1][j];
		}
		for (j=0; j < mlff_str->size_X2; j++){
			mlff_str->X2_traindataset[i][j] = mlff_str->X2_traindataset[i+1][j];
		}
	}

	for (j=0; j < mlff_str->size_X3; j++){
		mlff_str->X3_traindataset[mlff_str->n_cols-1][j] = 0.0;
	}
	for (j=0; j < mlff_str->size_X2; j++){
		mlff_str->X2_traindataset[mlff_str->n_cols-1][j] = 0.0;
	}

	int atom_typ = mlff_str->natm_typ_train[col_ID];
	mlff_str->natm_train_elemwise[atom_typ] = mlff_str->natm_train_elemwise[atom_typ] -1;
	mlff_str->natm_train_total = mlff_str->natm_train_total -1;

	for (i =col_ID; i < mlff_str->n_cols-1; i++){
		mlff_str->natm_typ_train[i] = mlff_str->natm_typ_train[i+1];
	}
	
	mlff_str->n_cols = mlff_str->n_cols - 1;
}


void get_N_r_hnl(SPARC_OBJ *pSPARC){
	int i, j, info;
	FILE *fp;
	char line[512];
	char a1[512], a2[512], a3[512], a4[512];
	int count1=0, count2=0;

	// fp = fopen("hnl.txt","r");
	fp = fopen(pSPARC->hnl_file_name,"r");

	fgets(line, sizeof (line), fp);
	sscanf(line, "%s%s%s%s", a1, a2, a3, a4);
	int N_r = atoi(a4);
	pSPARC->N_rgrid_MLFF = N_r;
	fclose(fp);
}
