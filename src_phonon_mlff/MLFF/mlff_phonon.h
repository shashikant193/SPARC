#ifndef PHONON_H
#define PHONON_H

#include "mlff_types.h"
#include "isddft.h"

void calculate_phonon(SPARC_OBJ *pSPARC, MLFF_Obj *mlff_str);
void get_SOAP(SPARC_OBJ *pSPARC, MLFF_Obj *mlff_str, SoapObj *soap_str, NeighList *nlist);

#endif
