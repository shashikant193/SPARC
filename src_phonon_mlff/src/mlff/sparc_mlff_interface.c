#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include <string.h>
#ifdef USE_MKL
    #define MKL_Complex16 double _Complex
    #include <mkl.h>
#else
    #include <cblas.h>
    #include <lapacke.h>
#endif

#include "isddft.h"
#include "electronicGroundState.h"
#include "initialization.h"
#include "ddbp_tools.h"
#include "md.h"
#include "tools_mlff.h"
#include "spherical_harmonics.h"
#include "soap_descriptor.h"
#include "mlff_types.h"
#include "linearsys.h"
#include "sparsification.h"
#include "regression.h"
#include "mlff_read_write.h"
#include "sparc_mlff_interface.h"

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))
#define au2GPa 29421.02648438959


/*
init_MLFF function initializes and dynamically allocates memory to various objects in MLFF_Obj

[Input]
1. soap_str: SOAP structure to the first MD structure
2. mlff_str: MLFF_Obj structure to be initialized
3. n_str_max: max reference structure in training dataset
4. n_train_max: max local descriptor per element type in training dataset
[Output]
1. mlff_str: MLFF_Obj structure initialized
*/

void init_MLFF(SPARC_OBJ *pSPARC) {

	// initialized the mlff structure within SPARC
	pSPARC->mlff_str = (MLFF_Obj *) malloc(sizeof(MLFF_Obj)*1);
	MLFF_Obj* mlff_str = pSPARC->mlff_str;

	int rank, nprocs;
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// formatting the hnl file name start
    char *INPUT_DIR    = malloc(L_PSD * sizeof(char));
    char *inpt_path = pSPARC->filename;
    char *pch = strrchr(inpt_path,'/'); // find last occurrence of '/'
    if (pch == NULL) { // in case '/' is not found
        snprintf(INPUT_DIR, L_PSD, "%s", ".");
    } else {
        memcpy(INPUT_DIR, inpt_path, pch-inpt_path);
        INPUT_DIR[(int)(pch-inpt_path)] = '\0';
    }


    char *merged_string    = malloc(L_PSD * sizeof(char));
    snprintf(merged_string, L_PSD, "%s%s%s", INPUT_DIR, "/",pSPARC->hnl_file_name);
    snprintf(pSPARC->hnl_file_name, L_STRING, "%s", merged_string);
    free(merged_string);
    merged_string    = malloc(L_PSD * sizeof(char));
    snprintf(merged_string, L_PSD, "%s%s%s", INPUT_DIR, "/",pSPARC->mlff_data_folder);
    snprintf(pSPARC->mlff_data_folder, L_STRING, "%s", merged_string);
    free(merged_string);
    // formatting the hnl file name end


	// initialized the MLFF output print names
	

	// read the hnl files
	if (rank==0){
		get_N_r_hnl(pSPARC);

		mlff_str->rgrid = (double *) malloc(sizeof(double)* pSPARC->N_rgrid_MLFF);
	 	mlff_str->h_nl = (double *) malloc(sizeof(double)* pSPARC->N_rgrid_MLFF* pSPARC->N_max_SOAP*(pSPARC->L_max_SOAP+1));
		mlff_str->dh_nl = (double *) malloc(sizeof(double)* pSPARC->N_rgrid_MLFF* pSPARC->N_max_SOAP*(pSPARC->L_max_SOAP+1));
		read_h_nl(pSPARC->N_max_SOAP, pSPARC->L_max_SOAP, mlff_str->rgrid, mlff_str->h_nl, mlff_str->dh_nl, pSPARC);
		
		MPI_Bcast(&pSPARC->N_rgrid_MLFF, 1, MPI_INT, 0, MPI_COMM_WORLD);

		MPI_Bcast(mlff_str->rgrid, pSPARC->N_rgrid_MLFF, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(mlff_str->h_nl, pSPARC->N_rgrid_MLFF* pSPARC->N_max_SOAP*(pSPARC->L_max_SOAP+1), MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(mlff_str->dh_nl, pSPARC->N_rgrid_MLFF* pSPARC->N_max_SOAP*(pSPARC->L_max_SOAP+1), MPI_DOUBLE, 0, MPI_COMM_WORLD);
	} else {
		MPI_Bcast(&pSPARC->N_rgrid_MLFF, 1, MPI_INT, 0, MPI_COMM_WORLD);
		mlff_str->rgrid = (double *) malloc(sizeof(double)* pSPARC->N_rgrid_MLFF);
	 	mlff_str->h_nl = (double *) malloc(sizeof(double)* pSPARC->N_rgrid_MLFF* pSPARC->N_max_SOAP*(pSPARC->L_max_SOAP+1));
		mlff_str->dh_nl = (double *) malloc(sizeof(double)* pSPARC->N_rgrid_MLFF* pSPARC->N_max_SOAP*(pSPARC->L_max_SOAP+1));
		
		MPI_Bcast(mlff_str->rgrid, pSPARC->N_rgrid_MLFF, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(mlff_str->h_nl, pSPARC->N_rgrid_MLFF* pSPARC->N_max_SOAP*(pSPARC->L_max_SOAP+1), MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(mlff_str->dh_nl, pSPARC->N_rgrid_MLFF* pSPARC->N_max_SOAP*(pSPARC->L_max_SOAP+1), MPI_DOUBLE, 0, MPI_COMM_WORLD);
	}

	// Number of stress components depedning on the BC
	int stress_len = (1-pSPARC->BCx) + (1-pSPARC->BCy) + (1-pSPARC->BCz) + ( (1-pSPARC->BCx-pSPARC->BCy) > 0) + ( (1-pSPARC->BCx-pSPARC->BCz) > 0) + ( (1-pSPARC->BCy-pSPARC->BCz) > 0);
	stress_len = (pSPARC->cell_typ > 20 && pSPARC->cell_typ < 30) ? 1 : stress_len;

#ifdef DEBUG
	if(!rank)
		printf("No. of physical stress components = %d\n", stress_len);
#endif

	mlff_str->stress_len = stress_len;
	mlff_str->mlff_flag = pSPARC->mlff_flag;
	int nelem = pSPARC->Ntypes;
	int natom = pSPARC->n_atom;
	
	int el_idx_global[natom];
	// element ID of all atoms
	int count = 0;
	for (int i=0; i < pSPARC->Ntypes; i++){
		for (int j=0; j < pSPARC->nAtomv[i]; j++){
			el_idx_global[count] = i;
			count++;
		}
	}


	mlff_str->Znucl = (int *)malloc(pSPARC->Ntypes*sizeof(int));
	for (int i=0; i <pSPARC->Ntypes; i++){
		mlff_str->Znucl[i] = pSPARC->Znucl[i];
	}


	// Atoms distributed to processors for MLFF parallelization
	int natom_per_procs_quot = natom/nprocs;
	int natom_per_procs_rem = natom%nprocs;

	int natom_procs[nprocs];
	for (int i=0; i < nprocs; i++){
		if (i < natom_per_procs_rem){
			natom_procs[i] = natom_per_procs_quot+1;
		} else {
			natom_procs[i] = natom_per_procs_quot;
		}
	}

	mlff_str->natom_domain = natom_procs[rank];
	mlff_str->atom_idx_domain = (int *)malloc(mlff_str->natom_domain*sizeof(int));
	mlff_str->el_idx_domain = (int *)malloc(mlff_str->natom_domain*sizeof(int));
	mlff_str->print_mlff_flag = pSPARC->print_mlff_flag;

	char str1[100] = "MLFF_data_reference_atoms.txt"; 
	snprintf(mlff_str->ref_atom_name, L_STRING, "%s%s%s", pSPARC->mlff_data_folder, "/",str1);
    // strcpy(mlff_str->ref_atom_name, str1);

    char str2[100] = "MLFF_data_reference_structures.txt"; 
    snprintf(mlff_str->ref_str_name, L_STRING, "%s%s%s", pSPARC->mlff_data_folder, "/",str2);
    // strcpy(mlff_str->ref_str_name, str2);

    char str3[100] = "MLFF_RESTART.txt"; 
    snprintf(mlff_str->restart_name, L_STRING, "%s%s%s", pSPARC->mlff_data_folder, "/",str3);
    // strcpy(mlff_str->restart_name, str3);

 

	int strt = 0;
	if (rank==0){
		strt = 0;
	} else{
		for (int i=0; i < rank; i++){
			strt += natom_procs[i];
		}
	}
	// atom index for the atoms in the processor domain
	for (int i= 0; i < mlff_str->natom_domain; i++){
		mlff_str->atom_idx_domain[i]  = strt +i;
		mlff_str->el_idx_domain[i] = el_idx_global[mlff_str->atom_idx_domain[i]];
	}


	int N_r = pSPARC->N_rgrid_MLFF;

	// size of the descriptor
	mlff_str->descriptor_typ = pSPARC->descriptor_typ_MLFF;
	int size_X2 = 0;
	int size_X3 = 0;

	if (pSPARC->descriptor_typ_MLFF==0){
		size_X2 = nelem * pSPARC->N_max_SOAP;
		size_X3 = ((nelem * pSPARC->N_max_SOAP+1)*(nelem * pSPARC->N_max_SOAP))/2 * (pSPARC->L_max_SOAP+1);
	}
	if (pSPARC->descriptor_typ_MLFF==1){
		size_X2 = pSPARC->N_max_SOAP;
		size_X3 = ((pSPARC->N_max_SOAP+1)*(pSPARC->N_max_SOAP))/2 * (pSPARC->L_max_SOAP+1);
	}
	
	// This stores the scaling factors and std dev of stress components
	mlff_str->relative_scale_stress = (double*) calloc(stress_len, sizeof(double));
	mlff_str->stress_scale = (double*) calloc(stress_len, sizeof(double));
	mlff_str->std_stress = (double*) calloc(stress_len, sizeof(double));

	mlff_str->condK_min = pSPARC->condK_min;
	mlff_str->if_sparsify_before_train = pSPARC->if_sparsify_before_train;
	mlff_str->n_str_max = pSPARC->n_str_max_mlff;
	mlff_str->n_train_max = pSPARC->n_train_max_mlff;
	mlff_str->n_str = 0;
	mlff_str->n_rows = 0;
	mlff_str->n_cols = 0;
	mlff_str->natm_train_total = 0;
	mlff_str->nelem = nelem;
	mlff_str->natom = natom;
	mlff_str->size_X2 = size_X2;
	mlff_str->size_X3 = size_X3;
	mlff_str->beta_2 = pSPARC->beta_2_SOAP;
	mlff_str->beta_3 = pSPARC->beta_3_SOAP;
	mlff_str->xi_3 = pSPARC->xi_3_SOAP;
	mlff_str->Nmax = pSPARC->N_max_SOAP;
	mlff_str->Lmax = pSPARC->L_max_SOAP;
	mlff_str->rcut = pSPARC->rcut_SOAP;
	mlff_str->F_tol = pSPARC->F_tol_SOAP;
	mlff_str->relative_scale_F = pSPARC->F_rel_scale;
    int BC[] = {pSPARC->BCx,pSPARC->BCy,pSPARC->BCz};
	int index[] = {0,1,2,3,4,5};

	reshape_stress(pSPARC->cell_typ,BC,index);
	for(int i = 0; i < stress_len; i++){
		mlff_str->relative_scale_stress[i] = pSPARC->stress_rel_scale[index[i]];
	}

	mlff_str->sigma_w = 100.0;
	mlff_str->sigma_v = 0.1;
	mlff_str->kernel_typ = pSPARC->kernel_typ_MLFF;
	mlff_str->error_train_E = 0.0;
	mlff_str->error_train_F = 0.0;
	mlff_str->error_train_stress = (double*) calloc(stress_len, sizeof(double));

	mlff_str->E_store_counter = 0;
	mlff_str->F_store_counter = 0;

	mlff_str->F_store = (double *) calloc(pSPARC->n_str_max_mlff*natom*3, sizeof(double));  // hardcoded here (use realloc in future)
	mlff_str->E_store = (double *) calloc(pSPARC->n_str_max_mlff, sizeof(double));       // hardcoded here
	mlff_str->stress_store = (double **) calloc(stress_len, sizeof(double*));

	for (int i = 0; i < stress_len; i++){
		mlff_str->stress_store[i] = (double *) calloc(pSPARC->n_str_max_mlff, sizeof(double)); 
	}

	// rgrid of the hnl for spline interpolation
	mlff_str->N_rgrid = pSPARC->N_rgrid_MLFF;

	// initialized the arrays to be used in regression
	mlff_str->cov_train = (double *) malloc(1*sizeof(double));
	mlff_str->AtA_SVD_U = (double *) malloc(1*sizeof(double));
	mlff_str->AtA_SVD_Vt = (double *) malloc(1*sizeof(double));
	mlff_str->AtA_SingVal = (double *) malloc(1*sizeof(double));
	mlff_str->AtA = (double *) malloc(1*sizeof(double));
	mlff_str->natm_train_elemwise = (int *) malloc(nelem * sizeof(int));
	for (int i=0; i<nelem; i++){
		mlff_str->natm_train_elemwise[i] = 0;
	}


	mlff_str->natm_typ_train = (int *)malloc(sizeof(int)*nelem * pSPARC->n_train_max_mlff);


	int K_size_row = pSPARC->n_str_max_mlff * (3*mlff_str->natom_domain + 1+stress_len);
	int K_size_column = nelem * pSPARC->n_train_max_mlff;
	int b_size = pSPARC->n_str_max_mlff*(3*mlff_str->natom_domain + 1+stress_len);
	int w_size = nelem * pSPARC->n_train_max_mlff;

	mlff_str->K_train = (double **) malloc(sizeof(double*)*K_size_row);
	for (int i =0; i < K_size_row; i++){
		mlff_str->K_train[i] = (double *) malloc(sizeof(double)*K_size_column);
	}

	mlff_str->b_no_norm = (double *) malloc(sizeof(double)*b_size);
	mlff_str->weights = (double *) malloc(sizeof(double)*w_size);

	for (int i = 0; i < K_size_row; i++){
		for (int j=0; j < K_size_column; j++){
			mlff_str->K_train[i][j] = 0.0;
		}
	}

	for (int i = 0; i < b_size; i++){
		mlff_str->b_no_norm[i] = 0.0;
	}

	for (int i = 0; i < w_size; i++){
		mlff_str->weights[i] = 0.0;
	}

	mlff_str->soap_descriptor_strdataset = (SoapObj *) malloc(sizeof(SoapObj)*pSPARC->n_str_max_mlff);

	mlff_str->X2_traindataset = (double **) malloc(sizeof(double*)*pSPARC->n_train_max_mlff*nelem);
	mlff_str->X3_traindataset = (double **) malloc(sizeof(double*)*pSPARC->n_train_max_mlff*nelem);
	for (int i = 0; i < pSPARC->n_train_max_mlff*nelem; i++){
		mlff_str->X2_traindataset[i] = (double *) malloc(sizeof(double)*size_X2);
		mlff_str->X3_traindataset[i] = (double *) malloc(sizeof(double)*size_X3);
	}

	mlff_str->n_train_max = pSPARC->n_train_max_mlff;	
	mlff_str->n_str_max  = pSPARC->n_str_max_mlff;


	

	char fname_mlff_print[100] = "mlff.log"; 
	char fname_mlff_print_loc[L_STRING];
	if (rank==0 && mlff_str->print_mlff_flag==1){
		snprintf(fname_mlff_print_loc, L_STRING, "%s%s%s", INPUT_DIR, "/", fname_mlff_print);
		mlff_str->fp_mlff = (FILE *) malloc(sizeof(FILE)*1);
		mlff_str->fp_mlff = fopen(fname_mlff_print_loc,"w");
	}
	free(INPUT_DIR);
	
}

/*
free_MLFF function frees the memory allocated to various objects in MLFF_Obj

[Input]
1. mlff_str: MLFF_Obj structure 
[Output]
1. mlff_str: MLFF_Obj structure 
*/

void free_MLFF(MLFF_Obj *mlff_str){

	int rank, nprocs;
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	

	if (rank==0 && mlff_str->print_mlff_flag==1){
		fclose(mlff_str->fp_mlff);
	}

	int natom = mlff_str->natom, K_size_row = mlff_str->n_str_max*(3*mlff_str->natom_domain + 1 + mlff_str->stress_len), nelem = mlff_str->nelem;

	free(mlff_str->Znucl);
	free(mlff_str->atom_idx_domain);
	free(mlff_str->el_idx_domain);
	free(mlff_str->E_store);
	free(mlff_str->F_store);
	for(int i = 0; i < mlff_str->stress_len; i++){
	 	free(mlff_str->stress_store[i]);
	}

	free(mlff_str->stress_store);
	free(mlff_str->cov_train);
	free(mlff_str->AtA_SVD_U);
	free(mlff_str->AtA_SVD_Vt);
	free(mlff_str->AtA_SingVal);
	free(mlff_str->AtA);
	free(mlff_str->natm_train_elemwise);
	free(mlff_str->natm_typ_train);
	for (int i = 0; i < (mlff_str->n_str_max * (3*mlff_str->natom_domain + 1+mlff_str->stress_len)); i++){
		free(mlff_str->K_train[i]);
	}

	free(mlff_str->K_train);
	free(mlff_str->b_no_norm);
	free(mlff_str->weights);
	if (mlff_str->mlff_flag == 1 || mlff_str->mlff_flag == 22){
		for (int i = 0; i < mlff_str->n_str; i++){
		 	delete_soapObj(mlff_str->soap_descriptor_strdataset+i, mlff_str->natom_domain);
		}
		free(mlff_str->soap_descriptor_strdataset);
	}
		

	for (int i=0; i < mlff_str->n_train_max*nelem; i++){
		free(mlff_str->X2_traindataset[i]);
		free(mlff_str->X3_traindataset[i]);
	}
	free(mlff_str->X2_traindataset);
	free(mlff_str->X3_traindataset);

	free(mlff_str->relative_scale_stress);
	free(mlff_str->stress_scale);
	free(mlff_str->std_stress);
	free(mlff_str->error_train_stress);
	free(mlff_str->rgrid);
	free(mlff_str->h_nl);
	free(mlff_str->dh_nl);
}


/*
MLFF_main function initializes the MLFF calculations, also sets up MLFF restarts

[Input]
1. pSPARC: SPARC structure
[Output]
1. pSPARC: SPARC structure 

Important variables:
pSPARC->mlff_flag==1  : Start on-the-fly from scratch (don't use any existing model)
pSPARC->mlff_flag==21 : Only use the trained MLFF model and do not train 
pSPARC->mlff_flag==22 : Continue training the existing MLFF model
*/

void MLFF_main(SPARC_OBJ *pSPARC){
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	double t1, t2, t3, t4;

t1 = MPI_Wtime();
	init_MLFF(pSPARC);
t2 = MPI_Wtime();
	
	MLFF_Obj* mlff_str = pSPARC->mlff_str;
	FILE *fp_mlff;
	if (pSPARC->print_mlff_flag == 1 && rank ==0){
		// fp_mlff = fopen("mlff.log","a");
		fp_mlff = mlff_str->fp_mlff;
		fprintf(fp_mlff,"--------------------------------------------------------------------------------------------\n");
		fprintf(fp_mlff, "MD step: 0\n");
	}



	if (pSPARC->print_mlff_flag == 1 && rank ==0){
		fprintf(fp_mlff, "MLFF object intialized. Time taken: %.3f s\n", t2-t1);
	}

	

	

	if (pSPARC->mlff_flag==1) {
		intialize_print_MLFF(mlff_str, pSPARC);
t1 = MPI_Wtime();
		Calculate_electronicGroundState(pSPARC);
t2 = MPI_Wtime();
		if (pSPARC->print_mlff_flag == 1 && rank ==0){
			fprintf(fp_mlff, "DFT call done for the first MD. Time taken: %.3f s\n", t2-t1);
		}

	    MPI_Bcast(pSPARC->forces, 3*pSPARC->n_atom, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	    MPI_Bcast(pSPARC->stress, 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		Initialize_MD(pSPARC);
		if(pSPARC->cell_typ != 0){
	    	coordinatetransform_map(pSPARC, pSPARC->n_atom, pSPARC->atom_pos);
	    }

t1 = MPI_Wtime();		
		sparc_mlff_interface_firstMD(pSPARC, mlff_str);
t2 = MPI_Wtime();
		if (pSPARC->print_mlff_flag == 1 && rank ==0){
			fprintf(fp_mlff, "Covariance matrices formed from first MD DFT data. Time taken: %.3f s\n", t2-t1);
		}

		if(pSPARC->cell_typ != 0){
			for(int i = 0; i < pSPARC->n_atom; i++)
	    		nonCart2Cart_coord(pSPARC, &pSPARC->atom_pos[3*i], &pSPARC->atom_pos[3*i+1], &pSPARC->atom_pos[3*i+2]);	
		}

	} else {
t1 = MPI_Wtime();	
		if (!rank){
			read_MLFF_files(mlff_str, pSPARC);
		}
t2 = MPI_Wtime();
		if (pSPARC->print_mlff_flag == 1 && rank ==0){
			fprintf(fp_mlff, "Existing MLFF ref-atom file read. Time taken: %.3f s\n", t2-t1);
		}
t1 = MPI_Wtime();
		MPI_Bcast(&mlff_str->n_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&mlff_str->n_str, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(mlff_str->weights, mlff_str->n_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(&mlff_str->mu_E, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(&mlff_str->std_E, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(&mlff_str->std_F, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		//MPI_Bcast(&mlff_str->mu_stress, 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(mlff_str->std_stress, mlff_str->stress_len, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(&mlff_str->E_scale, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(&mlff_str->F_scale, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(mlff_str->stress_scale, mlff_str->stress_len, MPI_DOUBLE, 0, MPI_COMM_WORLD); 

	
		MPI_Bcast(&mlff_str->n_str, 1, MPI_INT, 0, MPI_COMM_WORLD);

		
		MPI_Bcast(mlff_str->natm_train_elemwise, pSPARC->Ntypes, MPI_INT, 0, MPI_COMM_WORLD);

		
		MPI_Bcast(&mlff_str->n_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&mlff_str->natm_train_total, 1, MPI_INT, 0, MPI_COMM_WORLD);

		

		
		for (int i=0; i < mlff_str->n_cols; i++){
			MPI_Bcast(mlff_str->X2_traindataset[i], mlff_str->size_X2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			MPI_Bcast(mlff_str->X3_traindataset[i], mlff_str->size_X3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		}



		
		MPI_Bcast(&mlff_str->relative_scale_F, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(mlff_str->natm_typ_train, mlff_str->n_cols, MPI_INT, 0, MPI_COMM_WORLD);
t2 = MPI_Wtime();
		if (pSPARC->print_mlff_flag == 1 && rank ==0){
			fprintf(fp_mlff, "MLFF ref-atom read data broadcasted. Time taken: %.3f s\n", t2-t1);
		}



		if (pSPARC->mlff_flag==21){
t1 = MPI_Wtime();			
			MLFF_call_from_MD_only_predict(pSPARC, mlff_str);
t2 = MPI_Wtime();
			if (pSPARC->print_mlff_flag == 1 && rank ==0){
				fprintf(fp_mlff, "MLFF model prediction for E, F, stress done. Time taken: %.3f s\n", t2-t1);
			}
			if(pSPARC->cell_typ != 0){
	        	for(int i = 0; i < pSPARC->n_atom; i++)
	            	nonCart2Cart_coord(pSPARC, &pSPARC->atom_pos[3*i], &pSPARC->atom_pos[3*i+1], &pSPARC->atom_pos[3*i+2]);	
			}

			Initialize_MD(pSPARC);	
		} else if (pSPARC->mlff_flag ==22){
			char *fname_str;
			fname_str = (char *) malloc(sizeof(char)*512);
			char str1[512] = "MLFF_data_reference_structures.txt"; 
			strcpy(fname_str, str1);

			double **cell_data;
			double **LatUVec_data;
			double **apos_data; 
			double *Etot_data;
			double **F_data;
			double **stress_data;
			int *natom_data;
			int **natom_elem_data;

			natom_data = (int *) malloc(sizeof(int)*mlff_str->n_str);
			Etot_data = (double *) malloc(sizeof(double)*mlff_str->n_str);
			cell_data = (double **) malloc(sizeof(double*)*mlff_str->n_str);
			LatUVec_data = (double **) malloc(sizeof(double*)*mlff_str->n_str);
			stress_data = (double **) malloc(sizeof(double*)*mlff_str->n_str);
			natom_elem_data = (int **) malloc(sizeof(int*)*mlff_str->n_str);
			for (int i = 0; i < mlff_str->n_str; i++){
				cell_data[i] = (double *) malloc(sizeof(double)*3);
				LatUVec_data[i] = (double *) malloc(sizeof(double)*9);
				stress_data[i] = (double *) malloc(sizeof(double)*6);
				natom_elem_data[i] = (int *) malloc(sizeof(int)*pSPARC->Ntypes);
			}

			


t1 = MPI_Wtime();
			if (rank==0){
				apos_data = (double **) malloc(sizeof(double*)*mlff_str->n_str);
				F_data = (double **) malloc(sizeof(double*)*mlff_str->n_str);
				read_structures_MLFF_data(mlff_str->ref_str_name, mlff_str->n_str, pSPARC->Ntypes, cell_data, LatUVec_data, apos_data, Etot_data, F_data, stress_data, natom_data, natom_elem_data);
				if(pSPARC->cell_typ != 0){
					for (int istr = 0; istr < mlff_str->n_str; istr++){
						coordinatetransform_map(pSPARC, natom_data[istr], apos_data[istr]);
					}
				}
			}

t2 = MPI_Wtime();
			if (pSPARC->print_mlff_flag == 1 && rank ==0){
				fprintf(fp_mlff, "Read structures data from the exisiting MLFF model. Time taken: %.3f s\n", t2-t1);
			}

			


t1 = MPI_Wtime();
			pretrain_MLFF_model(mlff_str, pSPARC, cell_data, LatUVec_data, apos_data, Etot_data, F_data, stress_data, natom_data, natom_elem_data);
t2 = MPI_Wtime();
			if (pSPARC->print_mlff_flag == 1 && rank ==0){
				fprintf(fp_mlff, "Pretraining from the exisiting data done. Time taken: %.3f s\n", t2-t1);
			}


t1 = MPI_Wtime();
			Calculate_electronicGroundState(pSPARC);
t2 = MPI_Wtime();
			if (pSPARC->print_mlff_flag == 1 && rank ==0){
				fprintf(fp_mlff, "DFT call made after pretraining. Time taken: %.3f s\n", t2-t1);
			}
		    MPI_Bcast(pSPARC->forces, 3*pSPARC->n_atom, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		    MPI_Bcast(pSPARC->stress, 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		    

			Initialize_MD(pSPARC);
			if(pSPARC->cell_typ != 0){
	    		coordinatetransform_map(pSPARC, pSPARC->n_atom, pSPARC->atom_pos);
	   		}
			init_dyarray(&mlff_str->atom_idx_addtrain);
			for (int i = 0; i < pSPARC->n_atom; i++){
				append_dyarray(&(mlff_str->atom_idx_addtrain),i);
			}


t1 = MPI_Wtime();
			sparc_mlff_interface_initialMD(pSPARC, mlff_str);
t2 = MPI_Wtime();
			if (pSPARC->print_mlff_flag == 1 && rank ==0){
				fprintf(fp_mlff, "Covariance matrices formed after the DFT call. Time taken: %.3f s\n", t2-t1);
			}

			
			

t1 = MPI_Wtime();
			mlff_train_Bayesian(mlff_str);
t2 = MPI_Wtime();
			if (pSPARC->print_mlff_flag == 1 && rank ==0){
				fprintf(fp_mlff, "MLFF model trained with Bayesian regression. Time taken: %.3f s\n", t2-t1);
			}


			pSPARC->last_train_MD_iter = 0;
			delete_dyarray(&mlff_str->atom_idx_addtrain);



			// MPI_Barrier(MPI_COMM_WORLD);
			// exit(3);

			for (int i = 0; i < mlff_str->n_str-1; i++){
				if(rank == 0){
					free(apos_data[i]); // Check this for memory leak
					free(F_data[i]);   // Check this for memory leak
				}
				free(cell_data[i]);
				free(LatUVec_data[i]);
				free(stress_data[i]);
				free(natom_elem_data[i]);

			}

			
			if (rank == 0) {
				free(apos_data);   // Check this for memory leak
				free(F_data);     // Check this for memory leak	
			}

			free(cell_data);
			free(LatUVec_data);
			free(stress_data);
			free(natom_elem_data);

			free(Etot_data);
			free(natom_data);

			if(pSPARC->cell_typ != 0){
				for(int i = 0; i < pSPARC->n_atom; i++)
	    			nonCart2Cart_coord(pSPARC, &pSPARC->atom_pos[3*i], &pSPARC->atom_pos[3*i+1], &pSPARC->atom_pos[3*i+2]);	
			}

		}
	}  
	
	// if (pSPARC->print_mlff_flag == 1 && rank ==0){
	// 	// fprintf(fp_mlff,"--------------------------------------------------------------------------------------------\n");
	// 	fclose(fp_mlff);
	// }


}

/*
pretrain_MLFF_model function trains the MLFF model from the available SPARC-DFT data of energy, force and stress stored from previous on-the-fly run

[Input]
1. mlff_str: MLFF structure
2. pSPARC: SPARC structure
3. cell_data: lattice constant of all structures
4. apos_data: atom positions of all structures
5. Etot_data: Total energy of all structures
6. F_data: Forces in all structures
7. stress_data: stresss on al structures
8. natom_data: number of atoms on all structures
9. natom_elem_data: Number of atoms of each eleement type of all structures
[Output]
1. mlff_str: MLFF structure 

*/


void pretrain_MLFF_model(
	MLFF_Obj *mlff_str,
	SPARC_OBJ *pSPARC, 
	double **cell_data, 
	double **LatUVec_data,
	double **apos_data, 
	double *Etot_data, 
	double **F_data, 
	double **stress_data, 
	int *natom_data, 
	int **natom_elem_data)
{
	int rank, nprocs;
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	double t1, t2, t3, t4;

	FILE *fp_mlff;
	if (pSPARC->print_mlff_flag == 1 && rank ==0){
		// fp_mlff = fopen("mlff.log","a");
		fp_mlff = mlff_str->fp_mlff;
	}

	double cell[3], LatUVec[9], *apos, Etot, *F, *stress, *stress1;
	int *natom_elem, *atomtyp, count, natom_domain, *atom_idx_domain, *el_idx_domain;
	NeighList *nlist;
	SoapObj *soap_str;

	int n_str = mlff_str->n_str;
	mlff_str->n_str = 0;
	stress1 = (double *) malloc(sizeof(double)*mlff_str->stress_len);

	int *BC = (int*)malloc(3*sizeof(int));
	BC[0] = pSPARC->BCx;
	BC[1] = pSPARC->BCy;
	BC[2] = pSPARC->BCz;

	
	MPI_Bcast(natom_data, n_str, MPI_INT, 0, MPI_COMM_WORLD);
	
	mlff_str->n_rows = 0;
	for (int i = 0; i < n_str; i++){
		apos = (double *) malloc(sizeof(double)*3*natom_data[i]);
		F = (double *) malloc(sizeof(double)*3*natom_data[i]);
		stress = (double *) malloc(sizeof(double) * 6);
		nlist = (NeighList *) malloc(sizeof(NeighList)*1);
		soap_str = (SoapObj *) malloc(sizeof(SoapObj)*1);
		natom_elem = (int *) malloc(sizeof(int)*pSPARC->Ntypes);

		if (rank==0){
			cell[0] = cell_data[i][0];
			cell[1] = cell_data[i][1];
			cell[2] = cell_data[i][2];

			for (int j = 0; j < 9; j++){
				LatUVec[j] = LatUVec_data[i][j];
			}
			for (int j = 0; j < 3*natom_data[i]; j++){
				apos[j] = apos_data[i][j];
				F[j] = F_data[i][j];  // check
			}
			for (int j = 0; j < pSPARC->Ntypes; j++){
				natom_elem[j] = natom_elem_data[i][j];
			}
			Etot = Etot_data[i];
			for (int j = 0; j < 6; j++){
				stress[j] =  stress_data[i][j];
			}
		}
		MPI_Bcast(cell, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(LatUVec, 9, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(apos, 3*natom_data[i], MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(F, 3*natom_data[i], MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(stress, 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(&Etot, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(natom_elem, pSPARC->Ntypes, MPI_INT, 0, MPI_COMM_WORLD);


		// mlff_str->E_store[mlff_str->E_store_counter] = Etot/natom_data[i];
		// for (int j = 0; j < mlff_str->stress_len; j++){
		// 	mlff_str->stress_store[j][mlff_str->E_store_counter] = stress[j]/au2GPa;
		// }
		// for (int j=0; j < 3*natom_data[i]; j++){
		// 	mlff_str->F_store[mlff_str->F_store_counter+j] = F[j];
		// }
		// mlff_str->E_store_counter += 1;
		// mlff_str->F_store_counter += 3*natom_data[i];

		
		int index[6] = {0,1,2,3,4,5};
		reshape_stress(pSPARC->cell_typ, BC, index);
		for(int j = 0; j < 6; j++){
			stress[j] = stress[index[j]];
		}

		atomtyp = (int *) malloc(natom_data[i]*sizeof(int));

		count = 0;
		for (int ii=0; ii < pSPARC->Ntypes; ii++){
			for (int j=0; j < natom_elem[ii]; j++){
				atomtyp[count] = ii;
				count++;
			}
		}

		get_domain_decompose_mlff_natom(natom_data[i], pSPARC->Ntypes, pSPARC->nAtomv, nprocs, rank,  &natom_domain);

		atom_idx_domain = (int *)malloc(sizeof(int)*natom_domain);
		el_idx_domain = (int *)malloc(sizeof(int)*natom_domain);

		get_domain_decompose_mlff_idx(natom_data[i], pSPARC->Ntypes, pSPARC->nAtomv, nprocs, rank, natom_domain, atom_idx_domain, el_idx_domain);


		double *geometric_ratio = (double*) malloc(2 * sizeof(double));
    	geometric_ratio[0] = pSPARC->CUTOFF_y[0]/pSPARC->CUTOFF_x[0];
		geometric_ratio[1] = pSPARC->CUTOFF_z[0]/pSPARC->CUTOFF_x[0];

t1 = MPI_Wtime();
		build_nlist(mlff_str->rcut, pSPARC->Ntypes, natom_data[i], apos, atomtyp, pSPARC->cell_typ, BC, cell, LatUVec, pSPARC->twist, geometric_ratio, nlist, natom_domain, atom_idx_domain, el_idx_domain);
t2 = MPI_Wtime();
		if (pSPARC->print_mlff_flag == 1 && rank ==0){
			fprintf(fp_mlff, "Neighbor list done. Time taken: %.3f s\n", t2-t1);
		}

t1 = MPI_Wtime();
		build_soapObj(soap_str, nlist, mlff_str->rgrid, mlff_str->h_nl, mlff_str->dh_nl, apos, mlff_str->Nmax,
						 mlff_str->Lmax, mlff_str->beta_3, mlff_str->xi_3, pSPARC->N_rgrid_MLFF);

t2 = MPI_Wtime();
		if (pSPARC->print_mlff_flag == 1 && rank ==0){
			fprintf(fp_mlff, "SOAP descriptor done. Time taken: %.3f s\n", t2-t1);
		}
		for (int j = 0; j < mlff_str->stress_len; j++){
			stress1[j] = stress[j]/au2GPa;  // check this 
		}

		// fprintf(fp1, "before, std_stress: %f %f %f %f %f %f\n", mlff_str->std_stress[0],  mlff_str->std_stress[1], mlff_str->std_stress[2], mlff_str->std_stress[3], mlff_str->std_stress[4], mlff_str->std_stress[5]);
t1 = MPI_Wtime();
		add_newstr_rows(soap_str, nlist, mlff_str, Etot/natom_data[i], F, stress1);
t2 = MPI_Wtime();

		

		// fprintf(fp1, "after, std_stress: %f %f %f %f %f %f\n", mlff_str->std_stress[0],  mlff_str->std_stress[1], mlff_str->std_stress[2], mlff_str->std_stress[3], mlff_str->std_stress[4], mlff_str->std_stress[5]);
		if (pSPARC->print_mlff_flag == 1 && rank ==0){
			fprintf(fp_mlff, "Rows added for structure no. %d. Time taken: %.3f s\n", i+1, t2-t1);
		}

		free(apos);
		free(F);
		free(stress);
		free(atomtyp);
		free(natom_elem);
		free(geometric_ratio);
		
		

		clear_nlist(nlist, natom_domain);
		free(nlist);
		delete_soapObj(soap_str, natom_domain);
		free(soap_str);



		free(atom_idx_domain);
		free(el_idx_domain);
#ifdef DEBUG
		if (rank==0 ){
			printf("Added new structure # %d\n",i+1);
		}
#endif
	
	}
	free(stress1);
	free(BC);
	// if (pSPARC->print_mlff_flag == 1 && rank ==0){
	// 	fclose(fp_mlff);
	// }

}

/*
get_domain_decompose_mlff_natom function obtains the number of atoms for parallelization in MLFF

[Input]
1. natom: number of atom in the structure
2. nelem: number of element types in the structure
3. nAtomv: number of atom of different elements in the structure
4. nprocs: total number of processors
5. rank: rank for which the decomposition to be outputted
[Output]
1. natom_domain: number of atoms to be handled by the processor rank

*/

void get_domain_decompose_mlff_natom(
	int natom,
	int nelem,
	int *nAtomv,
	int nprocs,
	int rank,
	int *natom_domain)
{
	int natom_per_procs_quot = natom/nprocs;
	int natom_per_procs_rem = natom%nprocs;

	int el_idx_global[natom];

	// element ID of all atoms
	int count = 0;
	for (int i=0; i < nelem; i++){
		for (int j=0; j < nAtomv[i]; j++){
			el_idx_global[count] = i;
			count++;
		}
	}

	int natom_procs[nprocs];
	for (int i=0; i < nprocs; i++){
		if (i < natom_per_procs_rem){
			natom_procs[i] = natom_per_procs_quot+1;
		} else {
			natom_procs[i] = natom_per_procs_quot;
		}
	}

	*natom_domain = natom_procs[rank];
}

/*
get_domain_decompose_mlff_idx function obtains the idx of those atoms for parallelization in MLFF

[Input]
1. natom: number of atom in the structure
2. nelem: number of element types in the structure
3. nAtomv: number of atom of different elements in the structure
4. nprocs: total number of processors
5. rank: rank for which the decomposition to be outputted
6. natom_domain: number of atoms to be handled by the processor rank
[Output]
1. atom_idx_domain: index of atoms to be handled by the processor rank
2. el_idx_domain: index of element types of the atoms to be handled by the processor rank
*/

void get_domain_decompose_mlff_idx(
	int natom,
	int nelem,
	int *nAtomv,
	int nprocs,
	int rank,
	int natom_domain,
	int *atom_idx_domain,
	int *el_idx_domain)
{
	int natom_per_procs_quot = natom/nprocs;
	int natom_per_procs_rem = natom%nprocs;

	int el_idx_global[natom];

	// element ID of all atoms
	int count = 0;
	for (int i=0; i < nelem; i++){
		for (int j=0; j < nAtomv[i]; j++){
			el_idx_global[count] = i;
			count++;
		}
	}

	int natom_procs[nprocs];
	for (int i=0; i < nprocs; i++){
		if (i < natom_per_procs_rem){
			natom_procs[i] = natom_per_procs_quot+1;
		} else {
			natom_procs[i] = natom_per_procs_quot;
		}
	}

	int strt = 0;
	if (rank==0){
		strt = 0;
	} else{
		for (int i=0; i < rank; i++){
			strt += natom_procs[i];
		}
	}
	// atom index for the atoms in the processor domain
	for (int i= 0; i < natom_domain; i++){
		atom_idx_domain[i]  = strt +i;
		el_idx_domain[i] = el_idx_global[atom_idx_domain[i]];
	}
}

/*
MLFF_call function acts as the interface in the md.c file. This function is called if (mlff_idx>0) whenever the electronicgroundstate was called before

[Input]
1. pSPARC: SPARC object
[Output]
1. pSPARC: SPARC object
*/

void MLFF_call(SPARC_OBJ* pSPARC){
	// pSPARC->mlff_flag == 1 means on-the-fly MD from scratch, pSPARC->mlff_flag == 21 means only prediction using a known model, pSPARC->mlff_flag == 22 on-the-fly MD starting from a known model
	if (pSPARC->mlff_flag == 1 || pSPARC->mlff_flag == 22) {
		MLFF_call_from_MD(pSPARC, pSPARC->mlff_str);
	} else if (pSPARC->mlff_flag == 21) {
		MLFF_call_from_MD_only_predict(pSPARC, pSPARC->mlff_str);
	}

	if(pSPARC->cell_typ != 0){
		for(int i = 0; i < pSPARC->n_atom; i++)
	    	nonCart2Cart_coord(pSPARC, &pSPARC->atom_pos[3*i], &pSPARC->atom_pos[3*i+1], &pSPARC->atom_pos[3*i+2]);	
	}

}

/*
MLFF_call_from_MD function performs the on-the-fly MLFF (it has the logic implemented to skip DFT steps on the basis of UQ)

[Input]
1. pSPARC: SPARC object
2. mlff_str: MLFF object
[Output]
1. pSPARC: SPARC object
2. mlff_str: MLFF object
*/

void MLFF_call_from_MD(SPARC_OBJ *pSPARC, MLFF_Obj *mlff_str){

	double t1, t2, t3, t4;
	t3 = MPI_Wtime();

	int nprocs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    FILE *fp_mlff;
    if (pSPARC->print_mlff_flag == 1 && rank ==0){
    	// fp_mlff = fopen("mlff.log","a");
    	fp_mlff = mlff_str->fp_mlff;
    }
    
	int doMLFF = pSPARC->mlff_flag;
	int initial_train;
	if (pSPARC->mlff_flag==1){
		initial_train= pSPARC->begin_train_steps;
	}
	if (pSPARC->mlff_flag==22){
		initial_train = -1;

	}

	if (pSPARC->print_mlff_flag==1 && rank==0){
		fprintf(fp_mlff,"--------------------------------------------------------------------------------------------\n");
		fprintf(fp_mlff, "MD step: %d\n", pSPARC->MDCount);
	}

	int BC[] = {pSPARC->BCx, pSPARC->BCy, pSPARC->BCz};
	int index[] = {0,1,2,3,4,5};
	reshape_stress(pSPARC->cell_typ, BC, index);
	
	if (pSPARC->MDCount <= initial_train){
t1 = MPI_Wtime();		
		Calculate_electronicGroundState(pSPARC);
		if(pSPARC->cell_typ != 0){
			coordinatetransform_map(pSPARC, pSPARC->n_atom, pSPARC->atom_pos);
		}

t2 = MPI_Wtime();
		if (pSPARC->print_mlff_flag == 1 && rank ==0){
			fprintf(fp_mlff, "DFT call made. Time taken: %.3f s\n", t2-t1);
		}
		MPI_Bcast(pSPARC->forces, 3*pSPARC->n_atom, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(pSPARC->stress, 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		init_dyarray(&mlff_str->atom_idx_addtrain);
		for (int i = 0; i < pSPARC->n_atom; i++){
			append_dyarray(&(mlff_str->atom_idx_addtrain),i);
		}
t1 = MPI_Wtime();
		sparc_mlff_interface_initialMD(pSPARC, mlff_str);
t2 = MPI_Wtime();
		if (pSPARC->print_mlff_flag == 1 && rank ==0){
			fprintf(fp_mlff, "The covariance matrix updated. Time taken: %.3f s\n", t2-t1);
		} 
		if (pSPARC->MDCount == initial_train){
t1 = MPI_Wtime();
			mlff_train_Bayesian(mlff_str); 
t2 = MPI_Wtime();
			if (pSPARC->print_mlff_flag == 1 && rank ==0){
				fprintf(fp_mlff, "Bayesian linear regression done. Time taken: %.3f s\n", t2-t1);
			} 
			pSPARC->last_train_MD_iter = pSPARC->MDCount;
		}
		delete_dyarray(&mlff_str->atom_idx_addtrain);	
	} else {
		double *E_predict, *F_predict,  *bayesian_error, *stress_predict;
		E_predict = (double *)malloc(1*sizeof(double));
		F_predict = (double *)malloc(3*pSPARC->n_atom*sizeof(double));
		stress_predict = (double *)malloc(mlff_str->stress_len*sizeof(double));
		bayesian_error = (double *)malloc((3*pSPARC->n_atom+1+mlff_str->stress_len)*sizeof(double));
t1 = MPI_Wtime();
		sparc_mlff_interface_predict(pSPARC, mlff_str, E_predict, F_predict, stress_predict, bayesian_error); 
t2 = MPI_Wtime();
		if (pSPARC->print_mlff_flag == 1 && rank ==0){
			fprintf(fp_mlff, "MLFF prediction made for the current MD. Time taken: %.3f s\n", t2-t1);
		}
		double max_pred_error = fabs(largest(&bayesian_error[1+mlff_str->stress_len], 3*pSPARC->n_atom));
		if (pSPARC->MDCount == pSPARC->last_train_MD_iter + 1){
			pSPARC->F_tol_SOAP = pSPARC->factor_multiply_sigma_tol*max_pred_error;
		}

		if (pSPARC->print_mlff_flag == 1 && rank ==0){
			fprintf(fp_mlff, "max_pred_error: %.9E, F_tol_SOAP: %.9E.\n", max_pred_error, pSPARC->F_tol_SOAP);
		}
#ifdef DEBUG
		if (rank==0){
			printf("max_pred_error: %10.9f, pSPARC->F_tol_SOAP: %10.9f\n",max_pred_error,pSPARC->F_tol_SOAP);
		}
#endif		
		if (max_pred_error >= pSPARC->F_tol_SOAP){	
			if (pSPARC->print_mlff_flag == 1 && rank ==0){
				fprintf(fp_mlff, "Error prediction in ML Model is too large!\n");
			}
t1 = MPI_Wtime();
			Calculate_electronicGroundState(pSPARC);
			if(pSPARC->cell_typ != 0){
				coordinatetransform_map(pSPARC, pSPARC->n_atom, pSPARC->atom_pos);	
			}

t2 = MPI_Wtime();
			if (pSPARC->print_mlff_flag == 1 && rank ==0){
				fprintf(fp_mlff, "DFT call made. Time taken: %.3f s\n", t2-t1);
			}
			MPI_Bcast(pSPARC->forces, 3*pSPARC->n_atom, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			MPI_Bcast(pSPARC->stress, 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			double sum_square_error = 0.0;
			double max_force_error = 0.0;
// #ifdef DEBUG
// 			if (rank==0){
// 				// double au2GPa=29421.02648438959;
// 				printf("Error prediction in ML Model is too large!!\n");
// 				printf("Error Energy: %10.9f (Ha/atom)\n", fabs((1.0/pSPARC->n_atom)*pSPARC->Etot - E_predict[0]));

// 				if(pSPARC->BC == 2){
// 					for(int i = 0; i< mlff_str->stress_len; i++)
// 						printf("Stress[%d]: Error (GPa) = %10.9f, Error(percent) = %10.9f\n", index[i],fabs(stress_predict[i]- pSPARC->stress[index[i]]), fabs(stress_predict[i]- pSPARC->stress[index[i]])/fabs(pSPARC->stress[index[i]]) *100);
				
				
// 				} else {
// 					for(int i = 0; i< mlff_str->stress_len; i++)
// 						printf("Stress[%d]: Error (Ha/Bohr^n) = %10.9f, Error(percent) = %10.9f\n", index[i], fabs(stress_predict[i]-pSPARC->stress[index[i]]), fabs(stress_predict[i]-pSPARC->stress[index[i]])/fabs(pSPARC->stress[index[i]]) *100);
// 				}
// 				printf("Force Error atom-wise: \n");

				
// 				for (int i=0; i <pSPARC->n_atom; i++){
// 					sum_square_error += (pSPARC->forces[3*i]+F_predict[3*i])*(pSPARC->forces[3*i]+F_predict[3*i]);
// 					sum_square_error += (pSPARC->forces[3*i+1]+F_predict[3*i+1])*(pSPARC->forces[3*i+1]+F_predict[3*i+1]);
// 					sum_square_error += (pSPARC->forces[3*i+2]+F_predict[3*i+2])*(pSPARC->forces[3*i+2]+F_predict[3*i+2]);
					
// 					printf("%10.9f %10.9f %10.9f\n",pSPARC->forces[3*i]+F_predict[3*i],
// 					pSPARC->forces[3*i+1]+F_predict[3*i+1], pSPARC->forces[3*i+2]+F_predict[3*i+2]);

// 					if (fabs(pSPARC->forces[3*i]+F_predict[3*i]) > max_force_error){
// 						max_force_error = fabs(pSPARC->forces[3*i]+(F_predict[3*i]));
// 					}
// 					if (fabs(pSPARC->forces[3*i+1]+F_predict[3*i+1]) > max_force_error){
// 						max_force_error = fabs(pSPARC->forces[3*i+1]+(F_predict[3*i+1]));
// 					}
// 					if (fabs(pSPARC->forces[3*i+2]+F_predict[3*i+2]) > max_force_error){
// 						max_force_error = fabs(pSPARC->forces[3*i+2]+(F_predict[3*i+2]));
// 					}
// 				}
// 				printf("Error Force (max): %10.9f (Ha/Bohr)\n", max_force_error);
// 				printf("Error Force (RMS): %10.9f (Ha/Bohr)\n", sqrt(sum_square_error/(3*pSPARC->n_atom)));

// 				printf("Error Force Train (max): %10.9f (Ha/Bohr)\n", mlff_str->error_train_F);
// 				printf("Error Energy Train (max): %10.9f (Ha/atom)\n", mlff_str->error_train_E);
// 			}

// #endif
			if (pSPARC->print_mlff_flag == 1 && rank ==0){
				fprintf(fp_mlff, "Error Energy: %10.9f (Ha/atom)\n", fabs((1.0/pSPARC->n_atom)*pSPARC->Etot - E_predict[0]));

				if(pSPARC->BC == 2){
					for(int i = 0; i< mlff_str->stress_len; i++)
						fprintf(fp_mlff, "Stress[%d]: Error (GPa) = %10.9f, Error(percent) = %10.9f\n", index[i], fabs(stress_predict[i]- pSPARC->stress[index[i]])*au2GPa, fabs(stress_predict[i]- pSPARC->stress[index[i]])/fabs(pSPARC->stress[index[i]]) *100);
				
				
				} else {
					for(int i = 0; i< mlff_str->stress_len; i++)
						fprintf(fp_mlff, "Stress[%d]: Error (Ha/Bohr^n) = %10.9f, Error(percent) = %10.9f\n", index[i], fabs(stress_predict[i]-pSPARC->stress[index[i]]), fabs(stress_predict[i]-pSPARC->stress[index[i]])/fabs(pSPARC->stress[index[i]]) *100);
				}

				fprintf(fp_mlff, "Force Error atom-wise: \n");
				sum_square_error = 0.0;
				max_force_error = 0.0;
				for (int i=0; i <pSPARC->n_atom; i++){
					sum_square_error += (pSPARC->forces[3*i]+F_predict[3*i])*(pSPARC->forces[3*i]+F_predict[3*i]);
					sum_square_error += (pSPARC->forces[3*i+1]+F_predict[3*i+1])*(pSPARC->forces[3*i+1]+F_predict[3*i+1]);
					sum_square_error += (pSPARC->forces[3*i+2]+F_predict[3*i+2])*(pSPARC->forces[3*i+2]+F_predict[3*i+2]);
					
					fprintf(fp_mlff, "%10.9f %10.9f %10.9f\n",pSPARC->forces[3*i]+F_predict[3*i],
					pSPARC->forces[3*i+1]+F_predict[3*i+1], pSPARC->forces[3*i+2]+F_predict[3*i+2]);

					if (fabs(pSPARC->forces[3*i]+F_predict[3*i]) > max_force_error){
						max_force_error = fabs(pSPARC->forces[3*i]+(F_predict[3*i]));
					}
					if (fabs(pSPARC->forces[3*i+1]+F_predict[3*i+1]) > max_force_error){
						max_force_error = fabs(pSPARC->forces[3*i+1]+(F_predict[3*i+1]));
					}
					if (fabs(pSPARC->forces[3*i+2]+F_predict[3*i+2]) > max_force_error){
						max_force_error = fabs(pSPARC->forces[3*i+2]+(F_predict[3*i+2]));
					}
				}
				fprintf(fp_mlff, "Error Force (max): %10.9f (Ha/Bohr)\n", max_force_error);
				fprintf(fp_mlff, "Error Force (RMS): %10.9f (Ha/Bohr)\n", sqrt(sum_square_error/(3*pSPARC->n_atom)));

				fprintf(fp_mlff, "Error Force Train (max): %10.9f (Ha/Bohr)\n", mlff_str->error_train_F);
				fprintf(fp_mlff, "Error Energy Train (max): %10.9f (Ha/atom)\n", mlff_str->error_train_E);
			}

			// if (pSPARC->last_train_MD_iter - pSPARC->MDCount !=  -1){
			// 	pSPARC->last_train_MD_iter = pSPARC->MDCount;
			// }
			init_dyarray(&mlff_str->atom_idx_addtrain);
			for (int i = 0; i < pSPARC->n_atom; i++){
				if (bayesian_error[1+mlff_str->stress_len+3*i]>pSPARC->F_tol_SOAP || bayesian_error[1+mlff_str->stress_len+3*i+1]>pSPARC->F_tol_SOAP || bayesian_error[1+mlff_str->stress_len+3*i+2]>pSPARC->F_tol_SOAP){
					append_dyarray(&(mlff_str->atom_idx_addtrain),i);
				}
			}
t1 = MPI_Wtime();
			sparc_mlff_interface_initialMD(pSPARC, mlff_str);
t2 = MPI_Wtime();
			if (pSPARC->print_mlff_flag == 1 && rank ==0){
				fprintf(fp_mlff, "Covariance matrices updated! Time taken: %.3f s\n", t2-t1);
			}
t1 = MPI_Wtime();
			mlff_train_Bayesian(mlff_str); 
t2 = MPI_Wtime();
			if (pSPARC->print_mlff_flag == 1 && rank ==0){
				fprintf(fp_mlff, "Bayesian linear regression done! Time taken: %.3f s\n", t2-t1);
			}
			pSPARC->last_train_MD_iter = pSPARC->MDCount;
			delete_dyarray(&mlff_str->atom_idx_addtrain);
		} else {
#ifdef DEBUG
			// double au2GPa = 29421.02648438959;
			if (rank==0){
				printf("Skipping DFT at rank %d\n", rank);
			}
#endif
			if (pSPARC->print_mlff_flag == 1 && rank ==0){
				fprintf(fp_mlff, "Skipped DFT for this step!\n");
			}
			pSPARC->Etot = E_predict[0] * pSPARC->n_atom;
			for (int i=0; i < pSPARC->n_atom*3; i++){
				pSPARC->forces[i] = -1.0*F_predict[i];
			}

			for(int i = 0; i < mlff_str->stress_len; i++){
				pSPARC->stress[index[i]] = stress_predict[i];
				if(pSPARC->BC == 2)
					pSPARC->pres = -1.0/3.0*(pSPARC->stress[0]+pSPARC->stress[3]+pSPARC->stress[5]);
			}
			

			if (rank==0){
				write_MLFF_results(pSPARC);
			}
		}
		free(E_predict);
		free(F_predict);
		free(stress_predict);
		free(bayesian_error);
	}


t4 = MPI_Wtime();
// #ifdef DEBUG
// 	if (rank == 0) {
// 	    printf("MLFF_call_from_MD took %.3f s.\n", t4 - t3); 
// 	}
// #endif
	// if (pSPARC->print_mlff_flag==1 && rank==0){
	// 	// fprintf(fp_mlff,"--------------------------------------------------------------------------------------------\n");
	// 	fclose(fp_mlff);
	// }

}

/*
MLFF_call_from_MD_only_predict function performs MLFF caclulation only

[Input]
1. pSPARC: SPARC object
2. mlff_str: MLFF object
[Output]
1. pSPARC: SPARC object
2. mlff_str: MLFF object
*/
void MLFF_call_from_MD_only_predict(SPARC_OBJ *pSPARC, MLFF_Obj *mlff_str){
	int nprocs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double t1, t2;
    FILE *fp_mlff;
    if (pSPARC->print_mlff_flag==1 && rank==0){
    	// fp_mlff = fopen("mlff.log","a");
    	fp_mlff = mlff_str->fp_mlff;
    }

    if (pSPARC->print_mlff_flag==1 && rank==0){
		fprintf(fp_mlff,"--------------------------------------------------------------------------------------------\n");
		fprintf(fp_mlff, "MD step: %d\n", pSPARC->MDCount);
	}

	double *E_predict, *F_predict,  *bayesian_error, *stress_predict;
	E_predict = (double *)malloc(1*sizeof(double));
	F_predict = (double *)malloc(3*pSPARC->n_atom*sizeof(double));
	stress_predict = (double *)malloc(mlff_str->stress_len*sizeof(double));
	bayesian_error = (double *)malloc((3*pSPARC->n_atom+1+mlff_str->stress_len)*sizeof(double));

t1 = MPI_Wtime();
	sparc_mlff_interface_predict(pSPARC, mlff_str, E_predict, F_predict, stress_predict, bayesian_error); 
t2 = MPI_Wtime();
	if (pSPARC->print_mlff_flag == 1 && rank ==0){
		fprintf(fp_mlff, "Prediction from MLFF done! Time taken: %.3f s\n", t2-t1);
	}
	pSPARC->Etot = E_predict[0] * pSPARC->n_atom;
	for (int i=0; i < pSPARC->n_atom*3; i++){
		pSPARC->forces[i] = -1.0*F_predict[i];
	}

	int index[] = {0,1,2,3,4,5};
	int BC[] = {pSPARC->BCx, pSPARC->BCy, pSPARC->BCz};
	reshape_stress(pSPARC->cell_typ, BC, index);

	for(int i = 0; i < mlff_str->stress_len; i++){
		pSPARC->stress[index[i]] = stress_predict[i];
		if(pSPARC->BC == 2)
			pSPARC->pres = -1.0/3.0*(pSPARC->stress[0]+pSPARC->stress[3]+pSPARC->stress[5]);
	}

	free(E_predict);
	free(F_predict);
	free(bayesian_error);
	free(stress_predict);

	if (rank==0){
		write_MLFF_results(pSPARC);
	}
	if (pSPARC->print_mlff_flag==1 && rank==0){
		fprintf(fp_mlff,"--------------------------------------------------------------------------------------------\n");
		// fclose(fp_mlff);
	}
}




void sparc_mlff_interface_firstMD(SPARC_OBJ *pSPARC, MLFF_Obj *mlff_str){
double t1, t2, t3, t4;
t3 = MPI_Wtime();

	int rank, nprocs;
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
    FILE *fp_mlff;
    if (pSPARC->print_mlff_flag==1 && rank==0){
    	// fp_mlff = fopen("mlff.log","a");
    	fp_mlff = mlff_str->fp_mlff;
    }
	
	

	NeighList *nlist;
	SoapObj *soap_str;
	int *Z = (int *)malloc(pSPARC->Ntypes*sizeof(int));
	for (int i=0; i <pSPARC->Ntypes; i++){
		Z[i] = pSPARC->Znucl[i];
	}

	nlist = (NeighList *) malloc(sizeof(NeighList)*1);
	soap_str = (SoapObj *) malloc(sizeof(SoapObj)*1);

	double  F_estimate_max;
	double **K_predict, *K_predict_col_major;

	double *geometric_ratio = (double*) malloc(2 * sizeof(double));
	geometric_ratio[0] = pSPARC->CUTOFF_y[0]/pSPARC->CUTOFF_x[0];
	geometric_ratio[1] = pSPARC->CUTOFF_z[0]/pSPARC->CUTOFF_x[0];

	int *BC = (int*)malloc(3*sizeof(int));
	double *cell =(double *) malloc(3*sizeof(double));
	int *atomtyp = (int *) malloc(pSPARC->n_atom*sizeof(int));

	BC[0] = pSPARC->BCx;
	BC[1] = pSPARC->BCy;
	BC[2] = pSPARC->BCz;
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
	if (pSPARC->print_mlff_flag == 1 && rank ==0){
		fprintf(fp_mlff, "neighbor list and soap descriptor done! Time taken: %.3f s\n", t2-t1);
	}

// #ifdef DEBUG
// 	if (rank == 0) {
// 	    printf("nlist and soap took %.3f s.\n", t2 - t1); 
// 	}
// #endif
t1 = MPI_Wtime();

	double *force = (double *)malloc(3*pSPARC->n_atom*sizeof(double));
	for (int i=0; i < 3*pSPARC->n_atom; i++){
		force[i] = -1.0*pSPARC->forces[i];
	}
	double *stress = (double *)malloc(6*sizeof(double));
	int index[6] = {0,1,2,3,4,5};
	reshape_stress(pSPARC->cell_typ, BC, index);
	for(int i = 0; i < 6; i++)
		stress[i] = pSPARC->stress[index[i]];

	// printf("MLFF generate the design matrix for the first MD calling 'add_firstMD' rank : %d. . .\n", rank);
	add_firstMD(soap_str, nlist, mlff_str, pSPARC->Etot/pSPARC->n_atom, force, stress);

	if (rank==0){
		if(pSPARC->cell_typ != 0){
	        for(int i = 0; i < pSPARC->n_atom; i++)
	            nonCart2Cart_coord(pSPARC, &pSPARC->atom_pos[3*i], &pSPARC->atom_pos[3*i+1], &pSPARC->atom_pos[3*i+2]);	
		}
		print_new_ref_structure_MLFF(mlff_str, mlff_str->n_str, nlist, pSPARC->atom_pos, pSPARC->Etot, force, pSPARC->stress);
		if(pSPARC->cell_typ != 0){
	       coordinatetransform_map(pSPARC, pSPARC->n_atom, pSPARC->atom_pos);	
		}

	}

t2 = MPI_Wtime();
	
	if (pSPARC->print_mlff_flag == 1 && rank ==0){
		fprintf(fp_mlff, "Covariance matrix updated! Time taken: %.3f s\n", t2-t1);
		// fclose(fp_mlff);
	}
// #ifdef DEBUG
// 	if (rank == 0) {
// 	    printf("add_firstMD and print_new_ref_structure_MLFF took %.3f s.\n", t2 - t1); 
// 	}
// #endif
	free(geometric_ratio);
	free(BC);
	free(cell);
	free(atomtyp);
	delete_soapObj(soap_str, mlff_str->natom_domain);
	clear_nlist(nlist, mlff_str->natom_domain);
	free(nlist);
	free(soap_str);
	free(force);
	free(stress);
	free(Z);
t4 = MPI_Wtime();
// #ifdef DEBUG
// 	if (rank == 0) {
// 	    printf("sparc_mlff_interface_firstMD took %.3f s.\n", t4 - t3); 
// 	}
// #endif

}


void sparc_mlff_interface_initialMD(SPARC_OBJ *pSPARC, MLFF_Obj *mlff_str){

double t1, t2, t3, t4;
t3 = MPI_Wtime();
	
	int rank, nprocs;
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	FILE *fp_mlff;
    if (pSPARC->print_mlff_flag==1 && rank==0){
    	// fp_mlff = fopen("mlff.log","a");
    	fp_mlff = mlff_str->fp_mlff;
    }

	


	NeighList *nlist;
	SoapObj *soap_str;
	// printf("sparc_mlff_interface_initialMD calld in pSPARC->MDCount: %d\n", pSPARC->MDCount);
	int *Z;
	Z = (int *)malloc(pSPARC->Ntypes*sizeof(int));
	for (int i=0; i <pSPARC->Ntypes; i++){
		Z[i] = pSPARC->Znucl[i];
	}
	
	nlist = (NeighList *) malloc(sizeof(NeighList)*1);
	soap_str = (SoapObj *) malloc(sizeof(SoapObj)*1);

	int *BC, *atomtyp;
	double *cell, F_estimate_max;
	double **K_predict, *K_predict_col_major;
	int nelem = mlff_str->nelem;
	int size_X2 = mlff_str->size_X2, size_X3 = mlff_str->size_X3;
	double beta_2 = mlff_str->beta_2, beta_3 = mlff_str->beta_3, xi_3 = mlff_str->xi_3;

	double *geometric_ratio = (double*) malloc(2 * sizeof(double));
	geometric_ratio[0] = pSPARC->CUTOFF_y[0]/pSPARC->CUTOFF_x[0];
	geometric_ratio[1] = pSPARC->CUTOFF_z[0]/pSPARC->CUTOFF_x[0];

	BC = (int*)malloc(3*sizeof(int));
	cell =(double *) malloc(3*sizeof(double));
	atomtyp = (int *) malloc(pSPARC->n_atom*sizeof(int));

	BC[0] = pSPARC->BCx; BC[1] = pSPARC->BCy; BC[2] = pSPARC->BCz; // periodic
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
	if (pSPARC->print_mlff_flag == 1 && rank ==0){
		fprintf(fp_mlff, "neighbor list and soap descriptor done! Time taken: %.3f s\n", t2-t1);
	}



// #ifdef DEBUG
// 	if (rank == 0) {
// 	    printf("nlist and soap in sparc_mlff_interface_initialMD took %.3f s.\n", t2 - t1); 
// 	}
// #endif
t1 = MPI_Wtime();

	double *force = (double *)malloc(3*pSPARC->n_atom*sizeof(double));
	for (int i=0; i < 3*pSPARC->n_atom; i++){
		force[i] = -1.0*pSPARC->forces[i];
	}
	double *stress = (double *)malloc(6*sizeof(double));
	int index[6] = {0,1,2,3,4,5};
	reshape_stress(pSPARC->cell_typ, BC, index);
	for(int i = 0; i < 6; i++)
		stress[i] = pSPARC->stress[index[i]];

	add_newstr_rows(soap_str, nlist, mlff_str, pSPARC->Etot/pSPARC->n_atom, force, stress);



t2 = MPI_Wtime();
// #ifdef DEBUG
// 	if (rank == 0) {
// 	    printf("add_newstr_rows in sparc_mlff_interface_initialMD took %.3f s.\n", t2 - t1); 
// 	}
// #endif
	if (pSPARC->print_mlff_flag == 1 && rank ==0){
		fprintf(fp_mlff, "Covariance matrix rows updated! Time taken: %.3f s\n", t2-t1);
	}

	

t1 = MPI_Wtime();

	double *X2_gathered, *X3_gathered;

	X2_gathered = (double *) malloc(sizeof(double)*size_X2*soap_str->natom);
	X3_gathered = (double *) malloc(sizeof(double)*size_X3*soap_str->natom);

	double *X2_local, *X3_local;

	X2_local = (double *) malloc(sizeof(double)*size_X2*mlff_str->natom_domain);
	X3_local = (double *) malloc(sizeof(double)*size_X3*mlff_str->natom_domain);

	for (int i=0; i < mlff_str->natom_domain; i++){
		for (int j=0; j < size_X2; j++){
			X2_local[i*size_X2+j] = soap_str->X2[i][j];
		}
		for (int j=0; j < size_X3; j++){
			X3_local[i*size_X3+j] = soap_str->X3[i][j];
		}
	}

	int local_natoms[nprocs];
	MPI_Allgather(&mlff_str->natom_domain, 1, MPI_INT, local_natoms, 1, MPI_INT, MPI_COMM_WORLD);

	int recvcounts_X2[nprocs], recvcounts_X3[nprocs], displs_X2[nprocs], displs_X3[nprocs];
	displs_X2[0] = 0;
	displs_X3[0] = 0;
	for (int i=0; i < nprocs; i++){
		recvcounts_X2[i] = local_natoms[i]*size_X2;
		recvcounts_X3[i] = local_natoms[i]*size_X3;
		if (i>0){
			displs_X2[i] = displs_X2[i-1]+local_natoms[i-1]*size_X2;
			displs_X3[i] = displs_X3[i-1]+local_natoms[i-1]*size_X3;
		}
	}




	MPI_Allgatherv(X2_local, size_X2*mlff_str->natom_domain, MPI_DOUBLE, X2_gathered, recvcounts_X2, displs_X2, MPI_DOUBLE, MPI_COMM_WORLD);
	MPI_Allgatherv(X3_local, size_X3*mlff_str->natom_domain, MPI_DOUBLE, X3_gathered, recvcounts_X3, displs_X3, MPI_DOUBLE, MPI_COMM_WORLD);

	double **X2_gathered_2D, **X3_gathered_2D;
	X2_gathered_2D = (double **) malloc(sizeof(double*)*mlff_str->natom);
	X3_gathered_2D = (double **) malloc(sizeof(double*)*mlff_str->natom);
	for (int i=0; i < mlff_str->natom; i++){
		X2_gathered_2D[i] = (double *) malloc(sizeof(double)*size_X2);
		X3_gathered_2D[i] = (double *) malloc(sizeof(double)*size_X3);
		for (int j=0; j < size_X2; j++){
			X2_gathered_2D[i][j] = X2_gathered[i*size_X2+j];
		}
		for (int j=0; j < size_X3; j++){
			X3_gathered_2D[i][j] = X3_gathered[i*size_X3+j];
		}
	}


	double **X2_toadd, **X3_toadd;
	int no_desc_toadd[nelem];
	for (int i=0; i <nelem; i++){
		no_desc_toadd[i] = 0;
	}

	// atom_idx_addtrain stores the atom index whose force error predcition was larger 
	for (int i = 0; i <pSPARC->n_atom; i++){
		if (lin_search(((&mlff_str->atom_idx_addtrain))->array, ((&mlff_str->atom_idx_addtrain))->len, i) != -1){
			no_desc_toadd[atomtyp[i]] += 1;
		}
	}

	// Only atoms whose force error was larger than the threshold
	X2_toadd = (double **) malloc(sizeof(double *)* (((&mlff_str->atom_idx_addtrain))->len));
	X3_toadd = (double **) malloc(sizeof(double *)* (((&mlff_str->atom_idx_addtrain))->len));
	int atom_typ1[((&mlff_str->atom_idx_addtrain))->len];

	for (int i =0; i<(((&mlff_str->atom_idx_addtrain))->len); i++){
		X2_toadd[i] = (double *) malloc(sizeof(double)*size_X2);
		X3_toadd[i] = (double *) malloc(sizeof(double)*size_X3);
		atom_typ1[i] = atomtyp[((&mlff_str->atom_idx_addtrain))->array[i]];
		for (int j=0; j < size_X2; j++){
			X2_toadd[i][j] = X2_gathered_2D[((&mlff_str->atom_idx_addtrain))->array[i]][j];
		}
		for (int j=0; j < size_X3; j++){
			X3_toadd[i][j] = X3_gathered_2D[((&mlff_str->atom_idx_addtrain))->array[i]][j];
		}
	}

	int *cum_natm_elem;
	cum_natm_elem = (int *)malloc(sizeof(int)*nelem);
	cum_natm_elem[0] = 0;
	for (int i = 1; i < nelem; i++){
		cum_natm_elem[i] = cum_natm_elem[i-1]+no_desc_toadd[i-1];
	}

	dyArray *highrank_ID_descriptors;
	highrank_ID_descriptors = (dyArray *) malloc(sizeof(dyArray)*nelem);

	

	int total_descriptors_toadd = 0;
	for (int i=0; i <nelem; i++){
		
		int N_low_min = no_desc_toadd[i] - 500;
		init_dyarray(&highrank_ID_descriptors[i]);
		if (no_desc_toadd[i] > 0){
			SOAP_CUR_sparsify(mlff_str->kernel_typ, &X2_toadd[cum_natm_elem[i]], &X3_toadd[cum_natm_elem[i]],
				 no_desc_toadd[i], size_X2, size_X3, beta_2, beta_3, xi_3, &highrank_ID_descriptors[i], N_low_min);
		}
		
		total_descriptors_toadd += (highrank_ID_descriptors[i]).len;
	}


t2 = MPI_Wtime();
// #ifdef DEBUG
// 	if (rank == 0) {
// 	    printf("Sparsification in sparc_mlff_interface_initialMD took %.3f s.\n", t2 - t1); 
// 	}
// #endif
	if (pSPARC->print_mlff_flag == 1 && rank ==0){
		fprintf(fp_mlff, "Sparsification done! Time taken: %.3f s\n", t2-t1);
	}

t1 = MPI_Wtime();

	// int temp_idx;
	// for (int i=0; i < nelem; i++){
	// 	for (int j=0; j<(highrank_ID_descriptors[i]).len; j++){
	// 		temp_idx = cum_natm_elem[i] + (highrank_ID_descriptors[i]).array[j];
	// 		add_newtrain_cols(X2_toadd[temp_idx], X3_toadd[temp_idx], i, mlff_str);
	// 	}
	// }
	int temp_idx;

	for (int i=0; i < nelem; i++){
		for (int j=0; j<(highrank_ID_descriptors[i]).len; j++){
			
			temp_idx = cum_natm_elem[i] + (highrank_ID_descriptors[i]).array[j];

			add_newtrain_cols(X2_toadd[temp_idx], X3_toadd[temp_idx], i, mlff_str);

		}
	}



t2 = MPI_Wtime();
// #ifdef DEBUG
// if (rank == 0) {
// 	    printf("add_newtrain_cols in sparc_mlff_interface_initialMD took %.3f s.\n", t2 - t1); 
// 	}
// #endif

	if (pSPARC->print_mlff_flag == 1 && rank ==0){
		fprintf(fp_mlff, "Covariance matrix columns updated! Time taken: %.3f s\n", t2-t1);
	}

	if (rank==0){
		if(pSPARC->cell_typ != 0){
	        for(int i = 0; i < pSPARC->n_atom; i++)
	            nonCart2Cart_coord(pSPARC, &pSPARC->atom_pos[3*i], &pSPARC->atom_pos[3*i+1], &pSPARC->atom_pos[3*i+2]);	
		}
		print_new_ref_structure_MLFF(mlff_str, mlff_str->n_str, nlist, pSPARC->atom_pos, pSPARC->Etot, force, pSPARC->stress);
		if(pSPARC->cell_typ != 0){
	        coordinatetransform_map(pSPARC, pSPARC->n_atom, pSPARC->atom_pos);	
		}
	}


	
	for (int i=0; i <nelem; i++){
		delete_dyarray(&highrank_ID_descriptors[i]);
	}
	// if (pSPARC->print_mlff_flag == 1 && rank ==0){
	// 	fclose(fp_mlff);
	// }


	for (int i =0; i<(((&mlff_str->atom_idx_addtrain))->len); i++){
		free(X2_toadd[i]);
		free(X3_toadd[i]);
	}

	free(X2_toadd);
	free(X3_toadd); 
	free(cum_natm_elem);
	free(X2_local);
	free(X3_local);
	free(X2_gathered);
	free(X3_gathered);
	for (int i=0; i < mlff_str->natom; i++){
		free(X2_gathered_2D[i]);
		free(X3_gathered_2D[i]);
	}
	free(X2_gathered_2D);
	free(X3_gathered_2D);

	free(geometric_ratio);
	free(BC);
	free(cell);
	free(atomtyp);

	delete_soapObj(soap_str, mlff_str->natom_domain);

	clear_nlist(nlist, mlff_str->natom_domain);
	free(nlist);
	free(soap_str);
	free(force);
	free(stress);
	free(highrank_ID_descriptors);
	free(Z);

t4 = MPI_Wtime();
// #ifdef DEBUG
// 	if (rank == 0) {
// 	    printf("sparc_mlff_interface_initialMD took %.3f s.\n", t4 - t3); 
// 	}
// #endif
}

void sparc_mlff_interface_predict(SPARC_OBJ *pSPARC, MLFF_Obj *mlff_str, double *E_predict, double *F_predict,  double *stress_predict, double *bayesian_error){
double t1, t2, t3, t4;

t3 = MPI_Wtime();
	
	int nprocs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	FILE *fp_mlff;
    if (pSPARC->print_mlff_flag==1 && rank==0){
    	// fp_mlff = fopen("mlff.log","a");
    	fp_mlff = mlff_str->fp_mlff;
    }

	

	NeighList *nlist;
	SoapObj *soap_str;
	
	nlist = (NeighList *) malloc(sizeof(NeighList)*1);
	soap_str = (SoapObj *) malloc(sizeof(SoapObj)*1);

	int *BC, *atomtyp;
	double *cell, F_estimate_max;
	double **K_predict, *K_predict_col_major;

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
// #ifdef DEBUG
// 	if (rank == 0) {
// 	    printf("nlist and soap in sparc_mlff_interface_initialMD took %.3f s.\n", t2 - t1); 
// 	}
// #endif

	if (pSPARC->print_mlff_flag == 1 && rank ==0){
		fprintf(fp_mlff, "Neighbor list and soap descriptor done! Time taken: %.3f s\n", t2-t1);
	}


t1 = MPI_Wtime();

	if (rank==0){
		K_predict = (double **)malloc(sizeof(double*)*(1+3*mlff_str->natom_domain+mlff_str->stress_len));
		K_predict_col_major = (double *)malloc(sizeof(double)*(1+3*mlff_str->natom_domain+mlff_str->stress_len)*mlff_str->n_cols);
		for (int i=0; i<(1+3*mlff_str->natom_domain+mlff_str->stress_len); i++){
			K_predict[i] = (double *) malloc(sizeof(double)*mlff_str->n_cols);
			for (int j=0; j <mlff_str->n_cols; j++){
				K_predict[i][j] = 0.0;
			}
		}
	} else {
		K_predict = (double **)malloc(sizeof(double*)*(3*mlff_str->natom_domain));
		K_predict_col_major = (double *)malloc(sizeof(double)*(3*mlff_str->natom_domain)*mlff_str->n_cols);
		for (int i=0; i<(3*mlff_str->natom_domain); i++){
			K_predict[i] = (double *) malloc(sizeof(double)*mlff_str->n_cols);
			for (int j=0; j <mlff_str->n_cols; j++){
				K_predict[i][j] = 0.0;
			}
		}

	}

	calculate_Kpredict(soap_str, nlist, mlff_str, K_predict);
	
	if (rank==0){
		for (int j=0; j < mlff_str->n_cols; j++){
			for (int i=0; i<(1+3*mlff_str->natom_domain+mlff_str->stress_len); i++){
				K_predict_col_major[j*(1+3*mlff_str->natom_domain+mlff_str->stress_len)+i] = K_predict[i][j];
			}
		}
	} else {
		for (int j=0; j < mlff_str->n_cols; j++){
			for (int i=0; i<(3*mlff_str->natom_domain); i++){
				K_predict_col_major[j*(3*mlff_str->natom_domain)+i] = K_predict[i][j];
			}
		}
	}
	
	double *F_predict_local, *bayesian_error_local;

	F_predict_local = (double *) malloc(sizeof(double)*3*mlff_str->natom_domain);
	if (rank==0){
		bayesian_error_local = (double *) malloc(sizeof(double)*(1+3*mlff_str->natom_domain+mlff_str->stress_len));
	} else{
		bayesian_error_local = (double *) malloc(sizeof(double)*(3*mlff_str->natom_domain));
	}

t2 = MPI_Wtime();
// #ifdef DEBUG
// 	if (rank == 0) {
// 	    printf("calculate_Kpredict in sparc_mlff_interface_predict took %.3f s.\n", t2 - t1); 
// 	}
// #endif

	if (pSPARC->print_mlff_flag == 1 && rank ==0){
		fprintf(fp_mlff, "Kpredict calculation done done! Time taken: %.3f s\n", t2-t1);
	}
t1 = MPI_Wtime();
	

	mlff_predict(K_predict_col_major, mlff_str, E_predict, F_predict_local, stress_predict, bayesian_error_local, pSPARC->n_atom);
	MPI_Bcast(E_predict, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(stress_predict, mlff_str->stress_len, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	int local_natoms[nprocs];
	MPI_Allgather(&mlff_str->natom_domain, 1, MPI_INT, local_natoms, 1, MPI_INT, MPI_COMM_WORLD);

	int recvcounts[nprocs], displs[nprocs], recvcounts_bayesian[nprocs], displs_bayesian[nprocs];
	displs[0] = 0;
	displs_bayesian[0] = 0;

	for (int i=0; i < nprocs; i++){
		recvcounts[i] = local_natoms[i]*3;
		if (i==0){
			recvcounts_bayesian[i] = local_natoms[i]*3 + 1 + mlff_str->stress_len;
		} else {
			recvcounts_bayesian[i] = local_natoms[i]*3;
		}
		if (i>0){
			displs[i] = displs[i-1]+local_natoms[i-1]*3;
			if (i==1){
				displs_bayesian[i] = displs_bayesian[i-1]+local_natoms[i-1]*3+1+mlff_str->stress_len;
			} else {
				displs_bayesian[i] = displs_bayesian[i-1]+local_natoms[i-1]*3;
			}
		}
	}

	int send_bayesian;
	if (rank==0){
		send_bayesian = 3*mlff_str->natom_domain+1+mlff_str->stress_len;
	} else {
		send_bayesian = 3*mlff_str->natom_domain;
	}

	MPI_Allgatherv(F_predict_local, 3*mlff_str->natom_domain, MPI_DOUBLE, F_predict, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
	MPI_Allgatherv(bayesian_error_local, send_bayesian, MPI_DOUBLE, bayesian_error, recvcounts_bayesian, displs_bayesian, MPI_DOUBLE, MPI_COMM_WORLD);


	if (rank==0){
		for (int i=0; i<(1+mlff_str->stress_len+3*mlff_str->natom_domain); i++){
			free(K_predict[i]);
		}
	} else{
		for (int i=0; i<(3*mlff_str->natom_domain); i++){
			free(K_predict[i]);
		}
	}
t2 = MPI_Wtime();
// #ifdef DEBUG
// 	if (rank == 0) {
// 	    printf("mlff_predict with Allgather calls in sparc_mlff_interface_predict took %.3f s.\n", t2 - t1); 
// 	}
// #endif

	if (pSPARC->print_mlff_flag == 1 && rank ==0){
		fprintf(fp_mlff, "prediction and gather done! Time taken: %.3f s\n", t2-t1);
		// fclose(fp_mlff);
	}

	free(bayesian_error_local);
	free(F_predict_local);
	free(K_predict);
	free(K_predict_col_major);
	free(geometric_ratio);
	free(BC);
	free(cell);
	free(atomtyp);
	delete_soapObj(soap_str, mlff_str->natom_domain);

	clear_nlist(nlist, mlff_str->natom_domain);
	free(nlist);
	free(soap_str);
	free(Z);
t4 = MPI_Wtime();
// #ifdef DEBUG
// 	if (rank == 0) {
// 	    printf("sparc_mlff_interface_predict took %.3f s.\n", t4 - t3); 
// 	}
// #endif
}


void write_MLFF_results(SPARC_OBJ *pSPARC){
	

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
		fprintf(output_fp,"RMS force                          :%18.10E (Ha/Bohr)\n",avgF);
        fprintf(output_fp,"Maximum force                      :%18.10E (Ha/Bohr)\n",maxF);
        fclose(output_fp);
    }

    output_fp = fopen(pSPARC->OutFilename,"a");
    int stress_len = (1-pSPARC->BCx) + (1-pSPARC->BCy) + (1-pSPARC->BCz) + ( (1-pSPARC->BCx-pSPARC->BCy) > 0) + ( (1-pSPARC->BCx-pSPARC->BCz) > 0) + ( (1-pSPARC->BCy-pSPARC->BCz) > 0);
	stress_len = (pSPARC->cell_typ > 20 && pSPARC->cell_typ < 30) ? 1 : stress_len;
	double *stress = (double *)malloc(6*sizeof(double));
	int BC[] = {pSPARC->BCx, pSPARC->BCy, pSPARC->BCz};
	int index[6] = {0,1,2,3,4,5};
	reshape_stress(pSPARC->cell_typ, BC, index);
	for(int i = 0; i < 6; i++)
		stress[i] = pSPARC->stress[index[i]];
	
    double maxS = 0.0, temp;
    for (i = 0; i < stress_len; i++){
    	temp = fabs(stress[i]);
    	if(temp > maxS)
    		maxS = temp;
    }
    if (pSPARC->BC == 2){
        fprintf(output_fp,"Pressure                           :%18.10E (GPa)\n",-1.0/3.0*(pSPARC->stress[0]+pSPARC->stress[3]+pSPARC->stress[5])*au2GPa);
        fprintf(output_fp,"Maximum stress                     :%18.10E (GPa)\n",maxS*au2GPa);
    } else {
    	fprintf(output_fp,"Maximum Stress                     :%18.10E (Ha/Bohr^n) (n = periodicity order)\n",maxS);
    }
    fclose(output_fp);

    free(stress);
}


void reshape_stress(int cell_typ, int *BC, int *index) {

	int rank, nprocs;
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);


	int count = 0, skip_count=0;
	if(cell_typ < 20){
		for(int i = 0; i < 3; i++){
			for (int j = i; j < 3; j++){
				if(BC[i] == 0 && BC[j] == 0){
					index[count++] = 3*i+j-skip_count;
				}
			}
			skip_count = skip_count + i + 1;
		}
	} else{
		index[0] = 5;
	}
}



void coordinatetransform_map(SPARC_OBJ *pSPARC, int natom, double *coord) {
	// Convert Cart to nonCart coordinates for non orthogonal cell
    if(pSPARC->cell_typ != 0){
        for(int atm = 0; atm < natom; atm++)
	        Cart2nonCart_coord(pSPARC, &coord[3*atm], &coord[3*atm+1], &coord[3*atm+2]);
    }

}
