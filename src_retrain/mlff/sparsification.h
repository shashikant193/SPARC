#ifndef SPARSIFICATION_H
#define SPARSIFICATION_H

#include "mlff_types.h"
void SOAP_CUR_sparsify(int kernel_typ, double **X2, double **X3, int n_descriptor, 
	int size_X2, int size_X3, double beta_2, double beta_3, double xi_3, dyArray *highrank_ID_descriptors, int N_low_min);

#endif
