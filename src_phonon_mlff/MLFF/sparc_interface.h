#ifndef SPARC_INTERFACE_H
#define SPARC_INTERFACE_H

#include "isddft.h"
#include "mlff_types.h"
void sparc_mlff_interface_firstMD(SPARC_OBJ *pSPARC, MLFF_Obj *mlff_str);
void sparc_mlff_interface_initialMD(SPARC_OBJ *pSPARC, MLFF_Obj *mlff_str);
void sparc_mlff_interface_predict(SPARC_OBJ *pSPARC, MLFF_Obj *mlff_str, double *E_predict, double *F_predict, double *bayesian_error);
void write_MLFF_results(SPARC_OBJ *pSPARC);


#endif 