#ifndef SOAP_DESCRIPTOR_H
#define SOAP_DESCRIPTOR_H

#include "mlff_types.h"
#include "isddft.h"

void print_descriptors(double *X2, double *X3, double **dX2_dX, double **dX2_dY, double **dX2_dZ, double **dX2_dF, double **dX3_dF, double **dX3_dX, double **dX3_dY, double **dX3_dZ, int size_X2, int size_X3, int neighs);

void print_cnlm(double complex **cnlm, double complex ***dcnlm_dX, double complex ***dcnlm_dY, double complex ***dcnlm_dZ, double complex ***dcnlm_dF, int size_cnlm, int* neighs, int nelem);

void read_h_nl(const int N, const int L, double *rgrid, double *h_nl, double *dh_nl, double *d2h_nl, SPARC_OBJ *pSPARC);

void initialize_nlist(NeighList* nlist, int cell_typ, int *BC, double *cell_len, double *LatUVec, double twist, const int natom, const double rcut, const int nelem , const int natom_domain, int *atom_idx_domain, int *el_idx_domain);

int lin_search(int *arr, int n, int x);

void build_nlist(const double rcut, const int nelem, const int natom, const double * const atompos,
				 int * atomtyp, int cell_typ, int *BC, double *cell_len, double *LatUVec, double twist, double *geometric_ratio, NeighList* nlist, const int natom_domain, int *atom_idx_domain, int *el_idx_domain);

void clear_nlist(NeighList* nlist, int natom_domain);

int uniqueEle(int* a, int n);

void initialize_soapObj(SoapObj *soap_str, NeighList *nlist, int Lmax, int Nmax, int N_rgrid, double beta_3, double xi_3);

void build_soapObj(SoapObj *soap_str, NeighList *nlist, double* rgrid, double* h_nl, double* dh_nl, double *atompos, int Nmax, int Lmax, double beta_3, double xi_3, int N_rgrid);

void delete_soapObj(SoapObj *soap_str, int natom_domain);

void calculate_dtheta_dphi(double dx, double dy, double dz, double dr, double *dtheta, double *dphi, double *dth_dxi, double *dphi_dxi);

void calculate_rho2(double *r_ba, double hnl, double dhnl, double d2hnl, double complex Ylm, double complex *dYlm, double complex *d2Ylm, double complex *rho2);

void calculate_SOAP_d2X(SoapObj *soap_str, NeighList *nlist, double* rgrid, double* h_nl, double* dh_nl, double *d2h_nl, double *atompos, int Nmax, int Lmax, double beta_3, double xi_3, int N_rgrid, int Nq, double *qvec);

void free_soapObj_phonon(SoapObj *soap_str, int Nq);
void initialize_soapObj_phonon(SoapObj *soap_str, NeighList *nlist, int Nq);
double complex calc_phase_factor(NeighList *nlist, int i, int j, double *qvec);
#endif
