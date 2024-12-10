#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include "mkl.h"
#include "tools_mlff.h"
#include "spherical_harmonics.h"
#include "soap_descriptor.h"
#include "mlff_types.h"
#include "linearsys.h"
#include "sparsification.h"
#include "regression.h"
#include "isddft.h"
#include "electronicGroundState.h"
#include "MLFF_read_write.h"
#include "ddbp_tools.h"
#include "gmp_deriv_descriptor.h"

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))
#define au2GPa 29421.02648438959


void sparc_mlff_interface_firstMD(SPARC_OBJ *pSPARC, MLFF_Obj *mlff_str){

	printf("entered sparc_mlff_interface_firstMD\n");
	NeighList *nlist;
	
	int *Z;
	Z = (int *)malloc(pSPARC->Ntypes*sizeof(int));
	for (int i=0; i <pSPARC->Ntypes; i++){
		Z[i] = 1;//pSPARC->Znucl[i];
	}
	GMPObj *gmp_str;


	nlist = (NeighList *) malloc(sizeof(NeighList)*1);

	gmp_str = (GMPObj *) malloc(sizeof(GMPObj)*1);


	int *BC, *atomtyp;
	double *cell, F_estimate_max;
	double **K_predict, *K_predict_col_major, *force;


	// stress = (double *)malloc(6*sizeof(double));
	force = (double *)malloc(3*pSPARC->n_atom*sizeof(double));
	for (int i=0; i < 3*pSPARC->n_atom; i++)
		force[i] = -1.0*pSPARC->forces[i];


	BC = (int*)malloc(3*sizeof(int));
	cell =(double *) malloc(3*sizeof(double));
	atomtyp = (int *) malloc(pSPARC->n_atom*sizeof(int));

	BC[0] = 1; BC[1] = 1; BC[2] = 1;
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

	// stress[0] = pSPARC->stress[0];
	// stress[1] = pSPARC->stress[3];
	// stress[2] = pSPARC->stress[5];
	// stress[3] = pSPARC->stress[1];
	// stress[4] = pSPARC->stress[4];
	// stress[5] = pSPARC->stress[2];
	mlff_str->rcut = 35.0;
	initialize_nlist(nlist, pSPARC->n_atom, mlff_str->rcut, pSPARC->Ntypes);

	build_nlist(mlff_str->rcut, pSPARC->Ntypes, pSPARC->n_atom, pSPARC->atom_pos, atomtyp, BC, cell, nlist);
	printf("Building nlist done. \n");

	
	
	printf("Before initialize_gmpObj\n");

	initialize_gmpObj(gmp_str, nlist, mlff_str->cal_atoms, mlff_str->cal_num, mlff_str->params_i, 
		mlff_str->params_d, mlff_str->nmcsh, mlff_str->atom_gaussians_p, 
		mlff_str->ngaussians, mlff_str->element_index_to_order_p);
	printf("After initialize_gmpObj\n");
	FeatureScaler *ftr_scale;
	ftr_scale = (FeatureScaler *) malloc(sizeof(FeatureScaler)*1);
	ftr_scale->train = 1;



	build_gmpObj(gmp_str, nlist, ftr_scale, mlff_str->nmcsh, pSPARC->atom_pos, 
		mlff_str->params_i, mlff_str->params_d, mlff_str->atom_gaussians_p, 
		mlff_str->ngaussians, mlff_str->element_index_to_order_p, mlff_str->atom_type_to_indices_p, 
		mlff_str->atom_indices_p);

	


	// FILE *fp1;
	// fp1 = fopen("descriptor.txt","w");

	// for (int i = 0; i < 108; i++){
	// 	for (int j = 0; j < gmp_str->size_X; j++){
	// 		fprintf(fp1,"%.10f ",gmp_str->X[i][j]);
	// 	}
	// 	fprintf(fp1,"\n");
	// }
	// fclose(fp1);
	// exit(1);
	
	printf("After build_gmpObj\n");


	printf("MLFF generate the design matrix for the first MD calling 'add_firstMD' . . .\n");
	add_firstMD_GMP(gmp_str, nlist, mlff_str, pSPARC->Etot/pSPARC->n_atom, force);
	printf("Done: MLFF generate the design matrix for the first MD calling 'add_firstMD'\n");


	// for (int i=0; i < soap_str->natom; i++){
	// 	print_new_ref_atom_MLFF(mlff_str, atomtyp[i], 0, soap_str->X2[i], soap_str->X3[i]);
	// }

	// print_ref_atom_MLFF(mlff_str);

	print_new_ref_structure_MLFF(mlff_str, mlff_str->n_str, gmp_str, pSPARC->atom_pos, pSPARC->Etot, pSPARC->forces);
	// print_restart_MLFF(mlff_str, pSPARC, mlff_str->n_str+1, pSPARC->nAtomv);


	free(BC);
	free(cell);
	free(atomtyp);
	// free(stress);
	// if (pSPARC->descriptor_typ_MLFF==0){
	// 	delete_soapObj(soap_str);
	// } else if (pSPARC->descriptor_typ_MLFF==1){
	// 	delete_soapObj_wZ(soap_str);
	// }
	
	clear_nlist(nlist);
	free(nlist);

	free_GMP(gmp_str);
	free(gmp_str);


	// free(soap_str);
	free(force);
	free(Z);
	//free_GMP(gmp_str);
	//free(gmp_str);

	


}


void sparc_mlff_interface_initialMD(SPARC_OBJ *pSPARC, MLFF_Obj *mlff_str){

	NeighList *nlist;
	GMPObj *gmp_str;
	printf("sparc_mlff_interface_initialMD calld in pSPARC->MDCount: %d\n", pSPARC->MDCount);
	int *Z;
	Z = (int *)malloc(pSPARC->Ntypes*sizeof(int));
	for (int i=0; i <pSPARC->Ntypes; i++){
		Z[i] = 1;//pSPARC->Znucl[i];
	}
	
	nlist = (NeighList *) malloc(sizeof(NeighList)*1);
	gmp_str = (GMPObj *) malloc(sizeof(GMPObj)*1);

	int *BC, *atomtyp;
	double *cell, F_estimate_max;
	double **K_predict, *K_predict_col_major, *force;
	int nelem = mlff_str->nelem;
	int size_X = mlff_str->size_X_GMP;
	double xi_3 = mlff_str->xi_3;

	// stress = (double *)malloc(6*sizeof(double));
	


	


	BC = (int*)malloc(3*sizeof(int));
	cell =(double *) malloc(3*sizeof(double));
	atomtyp = (int *) malloc(pSPARC->n_atom*sizeof(int));

	BC[0] = 1; BC[1] = 1; BC[2] = 1; // periodic
	cell[0] = pSPARC->range_x;
	cell[1] = pSPARC->range_y;
	cell[2] = pSPARC->range_z;

	//  mlff stress stored in the order (xx,yy,zz,xy,yz,xz)
	// stress[0] = pSPARC->stress[0];
	// stress[1] = pSPARC->stress[3];
	// stress[2] = pSPARC->stress[5];
	// stress[3] = pSPARC->stress[1];
	// stress[4] = pSPARC->stress[4];
	// stress[5] = pSPARC->stress[2];

	int count = 0;
	for (int i=0; i < pSPARC->Ntypes; i++){
		for (int j=0; j < pSPARC->nAtomv[i]; j++){
			atomtyp[count] = i;
			count++;
		}
	}

	printf("Calling sparc_mlff_interface_initialMD\n");



	// printf("--------------------\n");
	// printf("Initializing nlist ... \n");
	mlff_str->rcut = 35.0;
	initialize_nlist(nlist, pSPARC->n_atom, mlff_str->rcut, pSPARC->Ntypes);
	// printf("Initialization of nlist done.\n");
	// printf("Building nlist ... \n");
	build_nlist(mlff_str->rcut, pSPARC->Ntypes, pSPARC->n_atom, pSPARC->atom_pos, atomtyp, BC, cell, nlist);
	// printf("Building nlist done. \n");
	// printf("Initializing soap ... \n");

	


	initialize_gmpObj(gmp_str, nlist, mlff_str->cal_atoms, mlff_str->cal_num, mlff_str->params_i, 
		mlff_str->params_d, mlff_str->nmcsh, mlff_str->atom_gaussians_p, 
		mlff_str->ngaussians, mlff_str->element_index_to_order_p);

	FeatureScaler *ftr_scale;
	ftr_scale = (FeatureScaler *) malloc(sizeof(FeatureScaler)*1);
	ftr_scale->train = 1;





	build_gmpObj(gmp_str, nlist, ftr_scale, mlff_str->nmcsh, pSPARC->atom_pos, 
		mlff_str->params_i, mlff_str->params_d, mlff_str->atom_gaussians_p, 
		mlff_str->ngaussians, mlff_str->element_index_to_order_p, mlff_str->atom_type_to_indices_p, 
		mlff_str->atom_indices_p);

	force = (double *)malloc(3*pSPARC->n_atom*sizeof(double));
	
	for (int i=0; i < 3*pSPARC->n_atom; i++)
		force[i] = -1.0*pSPARC->forces[i]; // negative because force -grad(E)

	for (int i = 0; i < mlff_str->size_X_GMP; i++){
		printf("%f\n",gmp_str->X[0][i]);
	}
	printf("gmp des printed\n");



	
	// printf("MLFF adding rows in the design matrix at pSPARC->MDCount: %d\n",pSPARC->MDCount);
	
	


	add_newstr_rows_GMP(gmp_str, nlist, mlff_str, pSPARC->Etot/pSPARC->n_atom, force);

	printf("Added the rows now doing CUR\n");

	
	// printf("Done: MLFF add rows in the design matrix at pSPARC->MDCount: %d\n",pSPARC->MDCount);
	// printf("MLFF adding columns in the design matrix at pSPARC->MDCountxxxx: %d\n",pSPARC->MDCount);
	
	// CUR sparsification (only removes the atoms from a given structure confiugration)
	double **X_toadd;
	int no_desc_toadd[nelem];
	for (int i=0; i <nelem; i++)
		no_desc_toadd[i] = 0;
	
	// atom_idx_addtrain stores the atom index whose force error predcition was larger 
	for (int i = 0; i <pSPARC->n_atom; i++){
		if (lin_search(((&mlff_str->atom_idx_addtrain))->array, ((&mlff_str->atom_idx_addtrain))->len, i) != -1){
			no_desc_toadd[atomtyp[i]] += 1;
		}
	}

	// Only atoms whose force error was larger than the threshold
	X_toadd = (double **) malloc(sizeof(double *)* (((&mlff_str->atom_idx_addtrain))->len));
	int atom_typ1[((&mlff_str->atom_idx_addtrain))->len];

	for (int i =0; i<(((&mlff_str->atom_idx_addtrain))->len); i++){
		X_toadd[i] = (double *) malloc(sizeof(double)*size_X);
		atom_typ1[i] = atomtyp[((&mlff_str->atom_idx_addtrain))->array[i]];
		for (int j=0; j < size_X; j++){
			X_toadd[i][j] = gmp_str->X[((&mlff_str->atom_idx_addtrain))->array[i]][j];
		}
	}

	int *cum_natm_elem;
	cum_natm_elem = (int *)malloc(sizeof(int)*nelem);
	cum_natm_elem[0] = 0;
	for (int i = 1; i < nelem; i++){
		cum_natm_elem[i] = no_desc_toadd[i-1];
	}

	dyArray *highrank_ID_descriptors;
	highrank_ID_descriptors = (dyArray *) malloc(sizeof(dyArray)*nelem);

	int total_descriptors_toadd = 0;
	for (int i=0; i <nelem; i++){
		init_dyarray(&highrank_ID_descriptors[i]);
		SOAP_CUR_sparsify_GMP(mlff_str->kernel_typ, &X_toadd[cum_natm_elem[i]],
				 no_desc_toadd[i], size_X, xi_3, &highrank_ID_descriptors[i]);
		total_descriptors_toadd += (highrank_ID_descriptors[i]).len;
	}
	int temp_idx;


	for (int i=0; i < nelem; i++) printf("%d\n",(highrank_ID_descriptors[i]).len);

	for (int i=0; i < nelem; i++){
		for (int j=0; j<(highrank_ID_descriptors[i]).len; j++){
			temp_idx = (highrank_ID_descriptors[i]).array[j];
			printf("i: %d, j: %d, temp_idx: %d, atom_typ1[temp_idx]: %d\n",i,j,temp_idx,atom_typ1[temp_idx]);
			add_newtrain_cols_GMP(X_toadd[temp_idx], atom_typ1[temp_idx], mlff_str);
		}
	}


	// for (int i=0; i < total_descriptors_toadd; i++){
		
	// 	add_newtrain_cols(X2_toadd[i], X3_toadd[i], atom_typ1[i], mlff_str);
	// }

	// for (int i=0; i < pSPARC->n_atom; i++){
	// 	add_newtrain_cols(soap_str->X2[i], soap_str->X3[i], atomtyp[i], mlff_str);
	// }
	printf("Done: MLFF add columns in the design matrix at pSPARC->MDCount: %d\n",pSPARC->MDCount);

	// for (int i=0; i < soap_str->natom; i++){
	// 	print_new_ref_atom_MLFF(mlff_str, atomtyp[i], 0, soap_str->X2[i], soap_str->X3[i]);
	// }
	// print_ref_atom_MLFF(mlff_str);


	print_new_ref_structure_MLFF(mlff_str, mlff_str->n_str, gmp_str, pSPARC->atom_pos, pSPARC->Etot, pSPARC->forces);

	for (int i=0; i <nelem; i++){
		delete_dyarray(&highrank_ID_descriptors[i]);
	}

	free(BC);
	free(cell);
	free(atomtyp);
	// free(stress);

	clear_nlist(nlist);
	free(nlist);
	free(force);
	free(highrank_ID_descriptors);
	free(Z);
	free_GMP(gmp_str);
	free(gmp_str);
}

void sparc_mlff_interface_predict(SPARC_OBJ *pSPARC, MLFF_Obj *mlff_str, double *E_predict, double *F_predict,  double *bayesian_error){

	NeighList *nlist;
	GMPObj *gmp_str;
	
	nlist = (NeighList *) malloc(sizeof(NeighList)*1);
	gmp_str = (GMPObj *) malloc(sizeof(GMPObj)*1);

	int *BC, *atomtyp;
	double *cell, F_estimate_max;
	double **K_predict, *K_predict_col_major;

	int *Z;
	Z = (int *)malloc(pSPARC->Ntypes*sizeof(int));
	for (int i=0; i <pSPARC->Ntypes; i++){
		Z[i] = 1;//pSPARC->Znucl[i];
	}

	BC = (int*)malloc(3*sizeof(int));
	cell =(double *) malloc(3*sizeof(double));
	atomtyp = (int *) malloc(pSPARC->n_atom*sizeof(int));

	BC[0] = 1; BC[1] = 1; BC[2] = 1;
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

	mlff_str->rcut = 35.0;
	initialize_nlist(nlist, pSPARC->n_atom, mlff_str->rcut, pSPARC->Ntypes);
	printf("Initialization of nlist done.\n");
	printf("Building nlist ... \n");
	build_nlist(mlff_str->rcut, pSPARC->Ntypes, pSPARC->n_atom, pSPARC->atom_pos, atomtyp, BC, cell, nlist);
	printf("Building nlist done. \n");
	printf("Initializing soap ... \n");



	printf("came here after soap\n");

	initialize_gmpObj(gmp_str, nlist, mlff_str->cal_atoms, mlff_str->cal_num, mlff_str->params_i, 
		mlff_str->params_d, mlff_str->nmcsh, mlff_str->atom_gaussians_p, 
		mlff_str->ngaussians, mlff_str->element_index_to_order_p);
	printf("After initialize_gmpObj\n");
	FeatureScaler *ftr_scale;
	ftr_scale = (FeatureScaler *) malloc(sizeof(FeatureScaler)*1);
	ftr_scale->train = 1;



	build_gmpObj(gmp_str, nlist, ftr_scale, mlff_str->nmcsh, pSPARC->atom_pos, 
		mlff_str->params_i, mlff_str->params_d, mlff_str->atom_gaussians_p, 
		mlff_str->ngaussians, mlff_str->element_index_to_order_p, mlff_str->atom_type_to_indices_p, 
		mlff_str->atom_indices_p);
	


	K_predict = (double **)malloc(sizeof(double*)*(1+3*pSPARC->n_atom));
	K_predict_col_major = (double *)malloc(sizeof(double)*(1+3*pSPARC->n_atom)*mlff_str->n_cols);
	for (int i=0; i<(1+3*pSPARC->n_atom); i++){
		K_predict[i] = (double *) malloc(sizeof(double)*mlff_str->n_cols);
		for (int j=0; j <mlff_str->n_cols; j++){
			K_predict[i][j] = 0.0;
		}
	}
	printf("Calling calculate_Kpredict\n");
	calculate_Kpredict_GMP(gmp_str, nlist, mlff_str, K_predict);
	printf("Returned calculate_Kpredict\n");
	FILE *fp1;
	fp1 = fopen("Kp.txt","w");
	for (int i=0; i < mlff_str->n_cols; i++){
		fprintf(fp1,"%f ",K_predict[0][i]);
	}
	fclose(fp1);
	
	

	for (int j=0; j < mlff_str->n_cols; j++)
		for (int i=0; i<(1+3*pSPARC->n_atom); i++)
			K_predict_col_major[j*(1+3*pSPARC->n_atom)+i] = K_predict[i][j];

	mlff_predict(K_predict_col_major, mlff_str, E_predict,  F_predict, bayesian_error, pSPARC->n_atom);

	for (int i=0; i<(1+3*pSPARC->n_atom); i++){
		free(K_predict[i]);
	}


	free(K_predict);
	free(K_predict_col_major);


	free(BC);
	free(cell);
	free(atomtyp);
	clear_nlist(nlist);
	free(nlist);
	free(Z);

	free_GMP(gmp_str);
	free(gmp_str);

}

void write_MLFF_results(SPARC_OBJ *pSPARC){
	
	printf("Calling write_MLFF_results\n");

	int rank, nproc, i;
	FILE *output_fp, *static_fp;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    // write energies into output file   

    if (!rank && pSPARC->Verbosity) {
        output_fp = fopen(pSPARC->OutFilename,"a");
	    if (output_fp == NULL) {
            printf("\nCannot open file \"%s\"\n",pSPARC->OutFilename);
            exit(EXIT_FAILURE);
        }
        fprintf(output_fp,"====================================================================\n");
        fprintf(output_fp,"                Energy and force calculation (MLFF #%d)                 \n", pSPARC->MDCount + pSPARC->restartCount + (pSPARC->RestartFlag == 0));
        fprintf(output_fp,"====================================================================\n");
        fprintf(output_fp,"Free energy per atom               :%18.10E (Ha/atom)\n", pSPARC->Etot / pSPARC->n_atom);
        fprintf(output_fp,"Total free energy                  :%18.10E (Ha)\n", pSPARC->Etot);
        fclose(output_fp);
        // for static calculation, print energy to .static file
        if (pSPARC->MDFlag == 0 && pSPARC->RelaxFlag == 0) {
            if (pSPARC->PrintForceFlag == 1 || pSPARC->PrintAtomPosFlag == 1) {
                static_fp = fopen(pSPARC->StaticFilename,"a");
                if (static_fp == NULL) {
                    printf("\nCannot open file \"%s\"\n",pSPARC->StaticFilename);
                    exit(EXIT_FAILURE);
                }
                fprintf(static_fp,"Total free energy from MLFF (Ha): %.15E\n", pSPARC->Etot);
                fclose(static_fp);
            }
        }
    }

    if(!rank && pSPARC->Verbosity) {
    	output_fp = fopen(pSPARC->OutFilename,"a");
        if (output_fp == NULL) {
            printf("\nCannot open file \"%s\"\n",pSPARC->OutFilename);
            exit(EXIT_FAILURE);
        }
        double avgF = 0.0, maxF = 0.0, temp;
        for (i = 0; i< pSPARC->n_atom; i++){
        	temp = fabs(sqrt(pow(pSPARC->forces[3*i  ],2.0) 
        	               + pow(pSPARC->forces[3*i+1],2.0) 
        	               + pow(pSPARC->forces[3*i+2],2.0)));
        	avgF += temp;
        	if (temp > maxF) maxF = temp;	
        }
        avgF /= pSPARC->n_atom;
		fprintf(output_fp,"RMS force (MLFF)                         :%18.10E (Ha/Bohr)\n",avgF);
        fprintf(output_fp,"Maximum force (MLFF)                     :%18.10E (Ha/Bohr)\n",maxF);
        fclose(output_fp);
    }

    output_fp = fopen(pSPARC->OutFilename,"a");
    double maxS = 0.0, temp;
    for (i = 0; i< 6; i++){
    	temp = fabs(pSPARC->stress[i]);
    	if(temp > maxS)
    		maxS = temp;	
    }
    // if (pSPARC->BC == 2){
    //     fprintf(output_fp,"Pressure                           :%18.10E (GPa)\n",-1.0/3.0*(pSPARC->stress[0]+pSPARC->stress[3]+pSPARC->stress[5])*au2GPa);
    //     fprintf(output_fp,"Maximum stress                     :%18.10E (GPa)\n",maxS*au2GPa);
    // }
    fclose(output_fp);


}
