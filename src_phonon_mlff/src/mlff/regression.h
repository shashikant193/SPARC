#ifndef REGRESSION_H
#define REGRESSION_H

#include "mlff_types.h"


void mlff_predict(double *K_predict, MLFF_Obj *mlff_str, double *E,  double* F, double* stress, double* error_bayesian, int natoms );
void CUR_sparsify_before_training(MLFF_Obj *mlff_str);
void mlff_train_Bayesian(MLFF_Obj *mlff_str);
void hyperparameter_Bayesian(double btb_reduced, double *AtA, double *Atb, MLFF_Obj *mlff_str, int M, double condK_min);
double get_regularization_min(double *A, int size, double condK_min);
#endif
