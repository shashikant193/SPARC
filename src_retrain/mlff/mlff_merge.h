#ifndef MLFF_MERGE_H
#define MLFF_MERGE_H

#include "mlff_types.h"
#include "isddft.h"

void MLFF_merge(SPARC_OBJ *pSPARC, MLFF_Obj *mlff_str);

void pretrain_MLFF_model_merge(
	MLFF_Obj *mlff_str,
	SPARC_OBJ *pSPARC, 
	int n_str,
	double **cell_data, 
	double **LatUVec_data,
	double **apos_data, 
	double *Etot_data, 
	double **F_data, 
	double **stress_data, 
	int *natom_data, 
	int **natom_elem_data);

void get_training_cols_numbers(char *folders_name, int *n_cols_model, int nelem);

int get_nstr_mlff_models(char *folders);

void get_training_cols_descriptors(char *folders, double ***X2_train_model, double ***X3_train_model, int nelem, int size_X2, int size_X3);

#endif