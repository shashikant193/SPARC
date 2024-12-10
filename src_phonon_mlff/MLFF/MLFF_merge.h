#ifndef MLFF_MERGE_H
#define MLFF_MERGE_H

#include "mlff_types.h"
#include "isddft.h"

void merge_mlff_atom_data(SPARC_OBJ *pSPARC);
void merge_mlff_only_str_data(SPARC_OBJ *pSPARC);
void add_newstr_rows_merge(SoapObj *soap_str, NeighList *nlist, MLFF_Obj *mlff_str, double E, double *F);
void mlff_train_Bayesian_merge(MLFF_Obj *mlff_str);
#endif
