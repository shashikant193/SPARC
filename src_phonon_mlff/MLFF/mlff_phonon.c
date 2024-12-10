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
#include "mlff_phonon.h"

void calculate_phonon(SPARC_OBJ *pSPARC, MLFF_Obj *mlff_str){

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double t1, t2;

    FILE *fp_mlff;
    if (pSPARC->print_mlff_flag == 1 && rank ==0){
        // fp_mlff = fopen("mlff.log","a");
        fp_mlff = mlff_str->fp_mlff;
    }

    t1 = MPI_Wtime();

    NeighList *nlist = (NeighList *) malloc(sizeof(NeighList)*1);
    SoapObj *soap_str = (SoapObj *) malloc(sizeof(SoapObj)*1);
    get_SOAP(pSPARC, mlff_str, soap_str, nlist);

    t2 = MPI_Wtime();

    if (pSPARC->print_mlff_flag == 1 && rank ==0){
        fprintf(fp_mlff, "get_SOAP calculation done. Time taken: %.3f s\n", t2-t1);
    }

    double complex weights[mlff_str->n_cols];

    for (int i = 0; i < mlff_str->n_cols; i++){
        weights[i] = mlff_str->weights[i] + 0.0*I;
    }

    
    int rows = 9*soap_str->natom*soap_str->natom;
    int cols = mlff_str->n_cols;

    t1 = MPI_Wtime();
    calculate_SOAP_d2X(soap_str, nlist, mlff_str->rgrid, mlff_str->h_nl, mlff_str->dh_nl, mlff_str->d2h_nl, pSPARC->atom_pos, mlff_str->Nmax,
                         mlff_str->Lmax, mlff_str->beta_3, mlff_str->xi_3, pSPARC->N_rgrid_MLFF, mlff_str->Nq, mlff_str->qvec);

    t2 = MPI_Wtime();
    if (pSPARC->print_mlff_flag == 1 && rank ==0){
        fprintf(fp_mlff, "calculate_SOAP_d2X calculation done. Time taken: %.3f s\n", t2-t1);
    }


    double complex **dyn_predict = (double _Complex**) malloc(sizeof(double _Complex*)*mlff_str->Nq); // row major;
    for (int i=0; i < mlff_str->Nq; i++){
        dyn_predict[i] = (double _Complex*) malloc(sizeof(double _Complex)*(9*soap_str->natom*soap_str->natom));
        for (int j = 0; j < (9*soap_str->natom*soap_str->natom); j++){
            dyn_predict[i][j] = 0.0+0.0*I;
        }
    }

    t1 = MPI_Wtime();
    calculate_Kpredict_phonon(soap_str, nlist, mlff_str, dyn_predict, weights);
    t2 = MPI_Wtime();
    if (pSPARC->print_mlff_flag == 1 && rank ==0){
        fprintf(fp_mlff, "calculate_Kpredict_phonon calculation done. Time taken: %.3f s\n", t2-t1);
    }
    
    
    
    t1 = MPI_Wtime();
    double phonon_eig[ mlff_str->Nq * 3*soap_str->natom];



    FILE *fp1,  *fp2;
    double complex alpha=1.0+0.0*I, beta = 0.0+0.0*I;

    if (rank ==0){
        printf("(pSPARC->amu2au *sqrt(pSPARC->Mass[ityp] * pSPARC->Mass[jtyp])): %.15f\n", (pSPARC->amu2au *sqrt(pSPARC->Mass[0] * pSPARC->Mass[0])));
        fp1 = fopen("dyn_mat.txt","w"); 

        printf("mlff_str->Nq: %d\n", mlff_str->Nq);
        for (int q = 0; q < mlff_str->Nq; q++){

            int count = 0;
            for (int ityp = 0; ityp < pSPARC->Ntypes; ityp++){
                 for (int i=0; i < 3*pSPARC->nAtomv[ityp]; i++){
                    for (int jtyp = 0; jtyp < pSPARC->Ntypes; jtyp++){
                        for (int j=0; j < 3*pSPARC->nAtomv[jtyp]; j++){
                            dyn_predict[q][count] /= (pSPARC->amu2au *sqrt(pSPARC->Mass[ityp] * pSPARC->Mass[jtyp]));
                            count++;
                        }
                        
                    }
                }
            }
            fprintf(fp1, "\n \n \n q: %d, qvec: %.15f %.15f %.15f\n", q, mlff_str->qvec[3*q], mlff_str->qvec[3*q+1], mlff_str->qvec[3*q+2]);
            for (int i = 0; i < 3*soap_str->natom; i++){
                for (int j = 0; j < 3*soap_str->natom; j++){
                    fprintf(fp1, "%.15f+%.15fi ", creal(dyn_predict[q][3*soap_str->natom *j +i]), cimag(dyn_predict[q][3*soap_str->natom *j +i]));
                }
                fprintf(fp1,"\n");
            }
            int info = LAPACKE_zheevd(LAPACK_ROW_MAJOR, 'V', 'U', 3*soap_str->natom, dyn_predict[q], 3*soap_str->natom, &(phonon_eig[3*soap_str->natom * q]));
        }

        fclose(fp1);
        printf("mlff_str->Nq: %d\n", mlff_str->Nq);
        fp2 = fopen("phonon.txt","w");
        for (int q = 0; q < mlff_str->Nq; q++){
            fprintf(fp2, "\n \n \nq: %d, qvec: %.15f %.15f %.15f\n", q, mlff_str->qvec[3*q], mlff_str->qvec[3*q+1], mlff_str->qvec[3*q+2]);

             fprintf(fp2,"\nphonon eigen-values, factor: %.15f\n", pSPARC->fs2atu * (1e5/(2.99792458)/2.0/M_PI));
            for (int i = 0; i < 3*soap_str->natom; i++){
                fprintf(fp2, "%.15f ", phonon_eig[3*soap_str->natom * q+i] );
                 // fprintf(fp2, "%.15f ", phonon_eig[3*soap_str->natom * q+i] );
            }
            fprintf(fp2,"\n phonon eigen-vectors\n");

            for (int i = 0; i < 3*soap_str->natom; i++){
                for (int j = 0; j < 3*soap_str->natom; j++){
                    // fprintf(fp2, "%.15f+%.15fi ", creal(dyn_predict[q][3*soap_str->natom *j +i]), cimag(dyn_predict[q][3*soap_str->natom *j +i]));
                    fprintf(fp2, "%.15f+%.15f ", creal(dyn_predict[q][3*soap_str->natom *j +i]), cimag(dyn_predict[q][3*soap_str->natom *j +i]));
                }
                fprintf(fp2,"\n");
            }
        }
        
        fclose(fp2);

    }

    t2 = MPI_Wtime();
    if (pSPARC->print_mlff_flag == 1 && rank ==0){
        fprintf(fp_mlff, "Eigensolve calculation done. Time taken: %.3f s\n", t2-t1);
    }



    
    free_soapObj_phonon(soap_str,  mlff_str->Nq);
    delete_soapObj(soap_str, mlff_str->natom_domain); 

    free(soap_str);

    clear_nlist(nlist, mlff_str->natom_domain);
    free(nlist);

    for (int i=0; i < mlff_str->Nq; i++){
        free(dyn_predict[i]);
    }
    free(dyn_predict);




    
}

void get_SOAP(SPARC_OBJ *pSPARC, MLFF_Obj *mlff_str, SoapObj *soap_str, NeighList *nlist){
    
    
    double t1, t2;
    int *BC, *atomtyp;
    double *cell;


    int *Z;
    Z = (int *)malloc(pSPARC->Ntypes*sizeof(int));
    for (int i=0; i <pSPARC->Ntypes; i++){
        Z[i] = pSPARC->Znucl[i];
    }

    double *geometric_ratio = (double*) malloc(2 * sizeof(double));
    geometric_ratio[0] = pSPARC->CUTOFF_y[0]/pSPARC->CUTOFF_x[0];
    geometric_ratio[1] = pSPARC->CUTOFF_z[0]/pSPARC->CUTOFF_x[0];

    BC = (int*)malloc(3*sizeof(int));
    cell =(double *) malloc(3*sizeof(double));
    atomtyp = (int *) malloc(pSPARC->n_atom*sizeof(int));

    BC[0] = pSPARC->BCx; BC[1] = pSPARC->BCy; BC[2] = pSPARC->BCz;
    cell[0] = pSPARC->range_x;
    cell[1] = pSPARC->range_y;
    cell[2] = pSPARC->range_z;

    int count = 0;
    for (int i=0; i < pSPARC->Ntypes; i++){
        for (int j=0; j < pSPARC->nAtomv[i]; j++){
            atomtyp[count] = i;
            count++;
        }
    }


t1 = MPI_Wtime();

    build_nlist(mlff_str->rcut, pSPARC->Ntypes, pSPARC->n_atom, pSPARC->atom_pos, atomtyp, pSPARC->cell_typ, BC, cell, pSPARC->LatUVec, pSPARC->twist, geometric_ratio, nlist, mlff_str->natom_domain, mlff_str->atom_idx_domain, mlff_str->el_idx_domain);


    build_soapObj(soap_str, nlist, mlff_str->rgrid, mlff_str->h_nl, mlff_str->dh_nl, pSPARC->atom_pos, mlff_str->Nmax,
                         mlff_str->Lmax, mlff_str->beta_3, mlff_str->xi_3, pSPARC->N_rgrid_MLFF);

t2 = MPI_Wtime();
    
}