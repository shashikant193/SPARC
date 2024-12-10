#ifndef LINEARSYS_H
#define LINEARSYS_H

#include "mlff_types.h"
#include "isddft.h"

double soap_kernel(int kernel_typ, double *X2_str, double *X3_str, double *X2_tr, double *X3_tr,
			 double beta_2, double beta_3, double xi_3, int size_X2, int size_X3);

double der_soap_kernel(int kernel_typ, double *dX2_str, double *dX3_str, double *X2_str, double *X3_str, double *X2_tr, double *X3_tr,
			 double beta_2, double beta_3, double xi_3, int size_X2, int size_X3);

double soap_kernel_polynomial(double *X2_str, double *X3_str, double *X2_tr, double *X3_tr,
			 double beta_2, double beta_3, double xi_3, int size_X2, int size_X3);

double soap_kernel_Gaussian(double *X2_str, double *X3_str, double *X2_tr, double *X3_tr,
			 double beta_2, double beta_3, double xi_3, int size_X2, int size_X3);

double soap_kernel_Laplacian(double *X2_str, double *X3_str, double *X2_tr, double *X3_tr,
			 double beta_2, double beta_3, double xi_3, int size_X2, int size_X3);

double der_soap_kernel_polynomial(double *dX2_str, double *dX3_str, double *X2_str, double *X3_str, double *X2_tr, double *X3_tr,
			 double beta_2, double beta_3, double xi_3, int size_X2, int size_X3);

double der_soap_kernel_Gaussian(double *dX2_str, double *dX3_str, double *X2_str, double *X3_str, double *X2_tr, double *X3_tr,
			 double beta_2, double beta_3, double xi_3, int size_X2, int size_X3);

double der_soap_kernel_Laplacian(double *dX2_str, double *dX3_str, double *X2_str, double *X3_str, double *X2_tr, double *X3_tr,
			 double beta_2, double beta_3, double xi_3, int size_X2, int size_X3);


void copy_descriptors(SoapObj *soap_str_MLFF, SoapObj *soap_str);

void add_firstMD(SoapObj *soap_str, NeighList *nlist, MLFF_Obj *mlff_str, double E, double* F, double* stress_sparc);

void add_newstr_rows(SoapObj *soap_str, NeighList *nlist, MLFF_Obj *mlff_str, double E, double *F, double* stress_sparc);

void calculate_Kpredict(SoapObj *soap_str, NeighList *nlist, MLFF_Obj *mlff_str, double **K_predict);

void add_newtrain_cols(double *X2, double *X3, int elem_typ, MLFF_Obj *mlff_str);

void remove_str_rows(MLFF_Obj *mlff_str, int str_ID);

void remove_train_cols(MLFF_Obj *mlff_str, int col_ID);

void get_N_r_hnl(SPARC_OBJ *pSPARC);

void calculate_Kpredict_phonon(SoapObj *soap_str, NeighList *nlist, MLFF_Obj *mlff_str, double complex **dyn_predict, double complex *weights);
double complex der2_soap_kernel_polynomial(double complex *dX3_str_alpha, double complex *dX3_str_beta,  double complex *d2X3_str, double complex *X3_str, double *X3_tr_d,
			  double xi_3, int size_X3);
double complex VectorDotProduct_complex_serial(const double complex *Vec1, const double complex *Vec2, const int len) ;
#endif
