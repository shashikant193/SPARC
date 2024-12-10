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
#include "mlff_merge.h"

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))
#define au2GPa 29421.02648438959


void MLFF_merge(SPARC_OBJ *pSPARC, MLFF_Obj *mlff_str){

	int rank, nprocs;
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	 // For merging of different models in folders locared in file folders_list.txt
	 // The first line of  'folders_list.txt' contains the number of models to be merged
	 // The subsequent lines contains the location of folders which contains those models



	mlff_str->stress_len = 1;
	FILE *fp_mlff;
    if (mlff_str->print_mlff_flag==1 && rank==0){
    	// fp_mlff = fopen("mlff.log","a");
    	fp_mlff = mlff_str->fp_mlff;
    }


	intialize_print_MLFF(mlff_str, pSPARC);



	char **folders;
	char line[1000];
	int folder_count;
	int *fcount_size;
	FILE *file;
	
	if (rank==0){
		
		file = fopen("folders_list.txt", "r");
	    if (fgets(line, sizeof(line), file) == NULL) {
	        perror("Error reading file!");
	        exit(1);
	    }
    	folder_count = strtoul(line, NULL, 10);
    	MPI_Bcast(&folder_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    	fcount_size = (int *) malloc(sizeof(int)*folder_count);
    	folders = (char **) malloc(folder_count * sizeof(char *));

    	for (int i = 0; i < folder_count; i++) {
    		if (fgets(line, sizeof(line), file) == NULL) {
	            perror("Error reading file!");
	            exit(1);
	        }
	        line[strcspn(line, "\n")] = '\0';
	        fcount_size[i] = 1000+1;
        	folders[i] = (char *) malloc(sizeof(char)*(1000+1));
	        strcpy(folders[i], line);
    	}
    	MPI_Bcast(fcount_size, folder_count, MPI_INT, 0, MPI_COMM_WORLD);
    	fclose(file);

    	
    	for (int i = 0; i < folder_count; i++) {
	        MPI_Bcast(folders[i], 1000 + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
    	}
	} else {
		MPI_Bcast(&folder_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

		folders = (char **) malloc(folder_count * sizeof(char *));
		fcount_size = (int *) malloc(sizeof(int)*folder_count);
		MPI_Bcast(fcount_size, folder_count, MPI_INT, 0, MPI_COMM_WORLD);

		for (int i = 0; i < folder_count; i++) {
			folders[i] = malloc(sizeof(char)*(1000+1));
			MPI_Bcast(folders[i], 1000+1, MPI_CHAR, 0, MPI_COMM_WORLD);
		}
	}






	int *n_cols_merged, *n_cols_model;
	n_cols_merged = (int *) calloc(pSPARC->Ntypes, sizeof(int));
	n_cols_model = (int *) calloc(pSPARC->Ntypes, sizeof(int));


	if (rank ==0){
		// n_cols_merged stores the total number of columns for a given element type
		for (int i = 0; i < folder_count; i++) {
			get_training_cols_numbers(folders[i], n_cols_model, pSPARC->Ntypes);
			for (int j = 0; j < pSPARC->Ntypes; j++){
				n_cols_merged[j] += n_cols_model[j];
			}
		}
	}


	
	

	MPI_Bcast(n_cols_merged, pSPARC->Ntypes, MPI_INT, 0, MPI_COMM_WORLD);


	
	

	// allocating memory for the training descriptors
	double ***X2_train_merged, ***X3_train_merged;

	X2_train_merged = (double ***) malloc(sizeof(double**) * pSPARC->Ntypes);
	X3_train_merged = (double ***) malloc(sizeof(double**) * pSPARC->Ntypes);
	for (int i = 0; i < pSPARC->Ntypes; i++){
		X2_train_merged[i] = (double **) malloc(sizeof(double*) * n_cols_merged[i]);
		X3_train_merged[i] = (double **) malloc(sizeof(double*) * n_cols_merged[i]);
		for (int j = 0; j < n_cols_merged[i]; j++){
			X2_train_merged[i][j] = (double *) malloc(sizeof(double) * mlff_str->size_X2);
			X3_train_merged[i][j] = (double *) malloc(sizeof(double) * mlff_str->size_X3);
		}
	}



	double ***X2_train_model, ***X3_train_model;

	int *count_des;
	count_des = (int *) calloc(pSPARC->Ntypes, sizeof(int));

	if (rank ==0){
		for (int i = 0; i < folder_count; i++) {
			get_training_cols_numbers(folders[i], n_cols_model, pSPARC->Ntypes);

			X2_train_model = (double ***) malloc(sizeof(double**) * pSPARC->Ntypes);
			X3_train_model = (double ***) malloc(sizeof(double**) * pSPARC->Ntypes);
			for (int j = 0; j < pSPARC->Ntypes; j++){
				X2_train_model[j] = (double **) malloc(sizeof(double*)* n_cols_model[j]);
				X3_train_model[j] = (double **) malloc(sizeof(double*)* n_cols_model[j]);
				for (int k = 0; k < n_cols_model[j]; k++){
					X2_train_model[j][k] = (double *) malloc(sizeof(double)* mlff_str->size_X2); 
					X3_train_model[j][k] = (double *) malloc(sizeof(double)* mlff_str->size_X3); 
				}
			}

			get_training_cols_descriptors(folders[i], X2_train_model, X3_train_model, pSPARC->Ntypes, mlff_str->size_X2, mlff_str->size_X3);

			for (int j = 0; j < pSPARC->Ntypes; j++) {
				for (int k = 0; k < n_cols_model[j]; k++){
					for (int l = 0; l < mlff_str->size_X2; l++){
						X2_train_merged[j][count_des[j]+k][l] = X2_train_model[j][k][l];
					}
					for (int l = 0; l < mlff_str->size_X3; l++){
						X3_train_merged[j][count_des[j]+k][l] = X3_train_model[j][k][l];
					}
				}
				count_des[j] +=  n_cols_model[j];
			}

			for (int j = 0; j < pSPARC->Ntypes; j++){
				for (int k = 0; k < n_cols_model[j]; k++){
					free(X2_train_model[j][k]);
					free(X3_train_model[j][k]);
				}
				free(X2_train_model[j]);
				free(X3_train_model[j]);
			}
			free(X2_train_model);
			free(X3_train_model);

		}
	}


	for (int i = 0; i < pSPARC->Ntypes; i++){
		for (int j = 0; j < n_cols_merged[i]; j++){
			MPI_Bcast(X2_train_merged[i][j], mlff_str->size_X2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			MPI_Bcast(X3_train_merged[i][j], mlff_str->size_X3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		}
	}

	
	// At this point we have ***X2_train_merged and ***X3_train_merged whose sizes are
	// [nelem][cols_elem][size_X2/size_X3]
	// and cols_elem is stored in n_cols_merged[elem]

	// copying  ***X2_train_merged and ***X3_train_merged into **mlff_str->X2_traindataset and **mlff_str->X2_traindataset
	// copying n_cols_merged[elem] into mlff_str->natm_train_total, mlff_str->natm_typ_train, mlff_str->natm_train_elemwise and mlff_str->n_cols


	mlff_str->natm_train_total = 0;
	mlff_str->n_cols = 0;



	// int count=0;
	// for (int i = 0; i < pSPARC->Ntypes; i++){
	// 	for (int j = 0; j < n_cols_merged[i]; j++){
	// 		for (int k = 0; k < mlff_str->size_X2; k++){
	// 			mlff_str->X2_traindataset[count][k] = X2_train_merged[i][j][k];
	// 		}
	// 		for (int k = 0; k < mlff_str->size_X3; k++){
	// 			mlff_str->X3_traindataset[count][k] = X3_train_merged[i][j][k];
	// 		}

	// 		mlff_str->natm_typ_train[count] = i;
	// 		mlff_str->natm_train_elemwise[i] += 1;

	// 		count++;

	// 		mlff_str->natm_train_total += 1;
	// 		mlff_str->n_cols += 1;
	// 	}
	// }


	// char fn[] = "desc.txt";
	// char new_fn[512];
	// sprintf(new_fn, "%s_%d", fn, rank);
	// FILE *fpn = fopen(new_fn,"w");


	// MPI_Barrier(MPI_COMM_WORLD);
	// exit(7);

	double g_ratio[14] = {0.099829919029546, 0.028778219799808,0.028137045849223,0.027777008252147,0.025457537079535,0.025069075226222,0.024533538408739,
						  0.021755504413314, 0.021363362212939, 0.021063412811737, 0.018995958225925, 0.017078281578255, 0.015513621989071, 0.014247215144346};

	// Reading the E, F and stress data from different folders and train the merged model
	char *fname_str;

	for (int idx_str = 0; idx_str < folder_count; idx_str++){



		int n_str = get_nstr_mlff_models(folders[idx_str]);
		char str1[512] = "MLFF_data_reference_structures.txt"; 

		fname_str = (char *) malloc(sizeof(char)*512);
		// strcpy(fname_str, folders[idx_str]);
		// strcat(fname_str, str1);

		sprintf(fname_str, "%s%s", folders[idx_str], str1);


		if (pSPARC->print_mlff_flag == 1 && rank ==0){
			fprintf(fp_mlff, "idx_str: %d\n", idx_str);
		}

		





		double **cell_data;
		double **LatUVec_data;
		double **apos_data; 
		double *Etot_data;
		double **F_data;
		double **stress_data;
		int *natom_data;
		int **natom_elem_data;


		natom_data = (int *) malloc(sizeof(int)*n_str);
		Etot_data = (double *) malloc(sizeof(double)*n_str);
		cell_data = (double **) malloc(sizeof(double*)*n_str);
		LatUVec_data = (double **) malloc(sizeof(double*)*n_str);
		stress_data = (double **) malloc(sizeof(double*)*n_str);
		natom_elem_data = (int **) malloc(sizeof(int*)*n_str);
		for (int i = 0; i < n_str; i++){
			cell_data[i] = (double *) malloc(sizeof(double)*3);
			LatUVec_data[i] = (double *) malloc(sizeof(double)*9);
			stress_data[i] = (double *) malloc(sizeof(double)*6);
			natom_elem_data[i] = (int *) malloc(sizeof(int)*pSPARC->Ntypes);
		}

		

		if (idx_str > 4){
			pSPARC->cell_typ=23;
			pSPARC->CyclixFlag = 1;
		}

		if (pSPARC->cell_typ==23 && n_str ==21){
			pSPARC->twist = 0.163682;
		} else if (pSPARC->cell_typ==23 && n_str != 21) {
			pSPARC->twist = 0.247336;
		}



		if (rank==0){
			apos_data = (double **) malloc(sizeof(double*)*n_str);
			F_data = (double **) malloc(sizeof(double*)*n_str);
			read_structures_MLFF_data(fname_str, n_str, pSPARC->Ntypes, cell_data, LatUVec_data, apos_data, Etot_data, F_data, stress_data, natom_data, natom_elem_data);
			
			if(pSPARC->cell_typ != 0){
				for (int istr = 0; istr < n_str; istr++){
					coordinatetransform_map(pSPARC, natom_data[istr], apos_data[istr]);
				}
			}

		}





		// printf("rank: %d, fname: %s, nstr: %d\n", rank, fname_str, n_str);
		// MPI_Barrier(MPI_COMM_WORLD);
		// exit(13);



		// printf("rank: %d, fname: %s\n",rank, fname_str);
		// MPI_Barrier(MPI_COMM_WORLD);
		// exit(15);

		

		pretrain_MLFF_model_merge(mlff_str, pSPARC, n_str, cell_data, LatUVec_data, apos_data, Etot_data, F_data, stress_data, natom_data, natom_elem_data);



		

		for (int i = 0; i < n_str; i++){
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


		free(fname_str);



		mlff_train_Bayesian(mlff_str);

		



	}

	

	mlff_train_Bayesian(mlff_str);


	MPI_Barrier(MPI_COMM_WORLD);
	exit(2);

	// Free all dynamic memory
	for (int i = 0; i < pSPARC->Ntypes; i++){
		for (int j = 0; j < n_cols_merged[i]; j++){
			free(X2_train_merged[i][j]);
			free(X3_train_merged[i][j]);
		}
		free(X2_train_merged[i]);
		free(X3_train_merged[i]);
	}
	free(X2_train_merged);
	free(X3_train_merged);

	free(n_cols_merged);
	free(n_cols_model);

	for (int i = 0; i < folder_count; i++){
		free(folders[i]);
	}
	free(folders);
	free(fcount_size);

	free(count_des);
}



/*
pretrain_MLFF_model_merge function trains the MLFF model from the available SPARC-DFT data of energy, force and stress stored from previous on-the-fly run

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
	int **natom_elem_data)
{
	int rank, nprocs;
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (pSPARC->cell_typ==23){
		printf("rank: %d, twist: %f\n", rank, pSPARC->twist);
	}
	

	if (pSPARC->cell_typ==23 && n_str ==21){
		pSPARC->twist = 0.163682;
	} else if (pSPARC->cell_typ==23 && n_str != 21) {
		pSPARC->twist = 0.247336;
	}

	if (pSPARC->cell_typ==23){
		printf("rank: %d, twist1: %f\n", rank, pSPARC->twist);
	}
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

	// int n_str = mlff_str->n_str;
	// mlff_str->n_str = 0;
	stress1 = (double *) malloc(sizeof(double)*mlff_str->stress_len);

	int *BC = (int*)malloc(3*sizeof(int));
	BC[0] = pSPARC->BCx;
	BC[1] = pSPARC->BCy;
	BC[2] = pSPARC->BCz;

	
	MPI_Bcast(natom_data, n_str, MPI_INT, 0, MPI_COMM_WORLD);


	
	// mlff_str->n_rows = 0;
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

		
		// for(int j = 0; j < 6; j++){
		// 	stress[j] = stress[index[j]];
		// }



		atomtyp = (int *) malloc(natom_data[i]*sizeof(int));

		count = 0;
		for (int ii=0; ii < pSPARC->Ntypes; ii++){
			for (int j=0; j < natom_elem[ii]; j++){
				atomtyp[count] = ii;
				count++;
			}
		}



		get_domain_decompose_mlff_natom(natom_data[i], pSPARC->Ntypes, natom_elem, nprocs, rank,  &natom_domain);

		atom_idx_domain = (int *)malloc(sizeof(int)*natom_domain);
		el_idx_domain = (int *)malloc(sizeof(int)*natom_domain);



		get_domain_decompose_mlff_idx(natom_data[i], pSPARC->Ntypes, natom_elem, nprocs, rank, natom_domain, atom_idx_domain, el_idx_domain);


		

		double *geometric_ratio = (double*) malloc(2 * sizeof(double));
    	// geometric_ratio[0] = 0.300042950522083;
		// geometric_ratio[1] = 1.0;//1.294218181818182;
		geometric_ratio[0] = pSPARC->CUTOFF_y[0]/pSPARC->CUTOFF_x[0];
		geometric_ratio[1] = pSPARC->CUTOFF_z[0]/pSPARC->CUTOFF_x[0];
		geometric_ratio[0] = 1;//;
		geometric_ratio[1] = 1;//1.206066541079478;
		// printf("Geno ratio: %.15f %.15f\n", geometric_ratio[0], geometric_ratio[1]);

t1 = MPI_Wtime();
		build_nlist(mlff_str->rcut, pSPARC->Ntypes, natom_data[i], apos, atomtyp, pSPARC->cell_typ, BC, cell, LatUVec, pSPARC->twist, geometric_ratio, nlist, natom_domain, atom_idx_domain, el_idx_domain);
t2 = MPI_Wtime();
		if (pSPARC->print_mlff_flag == 1 && rank ==0){
			fprintf(fp_mlff, "Neighbor list done. Time taken: %.3f s\n", t2-t1);
			// fprintf(fp_mlff, "geometric_ratio: %.15f %.15f\n", geometric_ratio[0], geometric_ratio[1]);
		}



t1 = MPI_Wtime();
		build_soapObj(soap_str, nlist, mlff_str->rgrid, mlff_str->h_nl, mlff_str->dh_nl, apos, mlff_str->Nmax,
						 mlff_str->Lmax, mlff_str->beta_3, mlff_str->xi_3, pSPARC->N_rgrid_MLFF);

		// if (rank == 0){
		// 	FILE *fp;
		// 	fp = fopen("desc.txt","w");
		// 	fprintf(fp, "natom_data[i]: %d\n",natom_data[i]);
		// 	for (int ix = 0; ix < 3*natom_data[i]; ix++){
		// 		fprintf(fp, "%f\n", apos[ix]);
		// 	}
		// 	fprintf(fp, "pSPARC->cell_typ: %d\n",pSPARC->cell_typ);
		// 	fprintf(fp, "BC: %d %d %d\n",BC[0], BC[1], BC[2]);
		// 	fprintf(fp, "CELL: %f %f %f\n",cell[0], cell[1], cell[2]);
		// 	fprintf(fp, "LatUVec: %f %f %f %f %f %f %f %f %f\n",LatUVec[0], LatUVec[1], LatUVec[2],
		// 						LatUVec[3], LatUVec[4], LatUVec[5], LatUVec[6], LatUVec[7], LatUVec[8]);
		// 	fprintf(fp, "pSPARC->twist: %f\n",pSPARC->twist);
		// 	fprintf(fp, "geometric_ratio: %f %f\n",geometric_ratio[0], geometric_ratio[1]);
		// 	fprintf(fp, "natom_domain: %d\n",natom_domain);
		// }
		// MPI_Barrier(MPI_COMM_WORLD);
		// exit(12);

t2 = MPI_Wtime();
		if (pSPARC->print_mlff_flag == 1 && rank ==0){
			fprintf(fp_mlff, "SOAP descriptor done. Time taken: %.3f s\n", t2-t1);
		}
		for (int j = 0; j < mlff_str->stress_len; j++){
			stress1[j] = stress[index[j]]/au2GPa;  // check this 
		}

		
		

t1 = MPI_Wtime();
	if (mlff_str->n_str == 0){
		add_firstMD(soap_str, nlist, mlff_str, Etot/natom_data[i], F, stress1);
	} else {
		add_newstr_rows(soap_str, nlist, mlff_str, Etot/natom_data[i], F, stress1);
	}
t2 = MPI_Wtime();
		
		// printf("ncols: %d, natom_elem[ii]: %d\n", mlff_str->n_cols, natom_elem[0]);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		int size_X2 = mlff_str->size_X2, size_X3 = mlff_str->size_X3;
		double *X2_gathered, *X3_gathered;
		X2_gathered = (double *) malloc(sizeof(double)*size_X2*soap_str->natom);
		X3_gathered = (double *) malloc(sizeof(double)*size_X3*soap_str->natom);

		double *X2_local, *X3_local;

		X2_local = (double *) malloc(sizeof(double)*size_X2*natom_domain);
		X3_local = (double *) malloc(sizeof(double)*size_X3*natom_domain);

		for (int ii=0; ii < natom_domain; ii++){
			for (int j=0; j < size_X2; j++){
				X2_local[ii*size_X2+j] = soap_str->X2[ii][j];
			}
			for (int j=0; j < size_X3; j++){
				X3_local[ii*size_X3+j] = soap_str->X3[ii][j];
			}
		}

		int local_natoms[nprocs];
		MPI_Allgather(&natom_domain, 1, MPI_INT, local_natoms, 1, MPI_INT, MPI_COMM_WORLD);

		int recvcounts_X2[nprocs], recvcounts_X3[nprocs], displs_X2[nprocs], displs_X3[nprocs];
		displs_X2[0] = 0;
		displs_X3[0] = 0;
		for (int ii=0; ii < nprocs; ii++){
			recvcounts_X2[ii] = local_natoms[ii]*size_X2;
			recvcounts_X3[ii] = local_natoms[ii]*size_X3;
			if (ii>0){
				displs_X2[ii] = displs_X2[ii-1]+local_natoms[ii-1]*size_X2;
				displs_X3[ii] = displs_X3[ii-1]+local_natoms[ii-1]*size_X3;
			}
		}

		MPI_Allgatherv(X2_local, size_X2*natom_domain, MPI_DOUBLE, X2_gathered, recvcounts_X2, displs_X2, MPI_DOUBLE, MPI_COMM_WORLD);
		MPI_Allgatherv(X3_local, size_X3*natom_domain, MPI_DOUBLE, X3_gathered, recvcounts_X3, displs_X3, MPI_DOUBLE, MPI_COMM_WORLD);

		double **X2_gathered_2D, **X3_gathered_2D;
		X2_gathered_2D = (double **) malloc(sizeof(double*)*soap_str->natom);
		X3_gathered_2D = (double **) malloc(sizeof(double*)*soap_str->natom);
		for (int ii=0; ii < soap_str->natom; ii++){
			X2_gathered_2D[ii] = (double *) malloc(sizeof(double)*size_X2);
			X3_gathered_2D[ii] = (double *) malloc(sizeof(double)*size_X3);
			for (int j=0; j < size_X2; j++){
				X2_gathered_2D[ii][j] = X2_gathered[ii*size_X2+j];
			}
			for (int j=0; j < size_X3; j++){
				X3_gathered_2D[ii][j] = X3_gathered[ii*size_X3+j];
			}
		}



		if (mlff_str->n_str > 0){
			for (int j=0; j < soap_str->natom; j++){
				add_newtrain_cols(X2_gathered_2D[j], X3_gathered_2D[j], 0, mlff_str);
			}
		}
		

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		

		

		// MPI_Barrier(MPI_COMM_WORLD);
		// exit(5);
		


		if (pSPARC->print_mlff_flag == 1 && rank ==0){
			fprintf(fp_mlff, "Rows added for structure no. %d. Time taken: %.3f s\n", i+1, t2-t1);
		}


		for (int j = 0; j < 6; j++){
			stress[j] = stress[j]/au2GPa;
		}
		if (rank==0){
			if(pSPARC->cell_typ != 0){
		        for(int j = 0; j < natom_data[i]; j++)
		            nonCart2Cart_coord(pSPARC, &apos[3*j], &apos[3*j+1], &apos[3*j+2]);	
			}

			print_new_ref_structure_MLFF(mlff_str, mlff_str->n_str, nlist, apos, Etot, F, stress);
			if(pSPARC->cell_typ != 0){
		        coordinatetransform_map(pSPARC, natom_data[i], apos);	
			}
		}

		// MPI_Barrier(MPI_COMM_WORLD);
		// exit(12);

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




void get_training_cols_numbers(char *folders_name, int *n_cols_model, int nelem){
	char str1[512] = "MLFF_RESTART.txt";

	char fname_str[1024];
	fname_str[0] = '\0';

	// strcpy(fname_str, folders_name);
	// strcpy(fname_str+, str1);

	sprintf(fname_str, "%s%s", folders_name, str1);

	FILE *file = fopen(fname_str, "r");

	char line[256];
    const char *target = "N_ref_atoms_elemwise:";
    int found = 0;
    int count = 0;




    while (fgets(line, sizeof(line), file)) {
        if (found) {
            if (sscanf(line, "%d", &n_cols_model[count]) == 1) {
                count++;
            } else {
                break; // Stop reading numbers when a non-number line is encountered
            }
        }
        if (strstr(line, target)) {
            found = 1; // Start reading numbers from the next line
        }
    }
    fclose(file);

}

int get_nstr_mlff_models(char *folder){

	int nstr;

	char str1[512] = "MLFF_RESTART.txt";

	char fname_str[1024];
	fname_str[0] = '\0';

	// strcpy(fname_str, folder);
	// strcpy(fname_str, str1);

	sprintf(fname_str, "%s%s", folder, str1);

	FILE *file = fopen(fname_str, "r");

	char line[256];
    const char *target = "N_ref_str:";
    int found = 0;
    int count = 0;

    while (fgets(line, sizeof(line), file)) {
        if (found) {
            if (sscanf(line, "%d", &nstr) == 1) {
                count++;
            } else {
                break; // Stop reading numbers when a non-number line is encountered
            }
        }
        if (strstr(line, target)) {
            found = 1; // Start reading numbers from the next line
        }
    }
    fclose(file);

    return nstr;
}

void get_training_cols_descriptors(char *folder, double ***X2_train_model, double ***X3_train_model, int nelem, int size_X2, int size_X3){
	int nstr;

	char str1[512] = "MLFF_data_reference_atoms.txt";

	char fname_str[1024];
	fname_str[0] = '\0';

	char a1[512], str[512];

	// strcpy(fname_str, folder);
	// strcpy(fname_str, str1);

	sprintf(fname_str, "%s%s", folder, str1);

	FILE *fptr = fopen(fname_str, "r");


	fgets(a1, sizeof (a1), fptr);
    for (int i=0; i < nelem; i++){
        fgets(a1, sizeof (a1), fptr);
    }

    int atm_typ, nimg;
    int img_no;
    double wt_temp;
    int temp_int;


    int count[nelem];

    for (int i = 0; i < nelem; i++){
    	count[i] = 0;
    }

    int elem_type;
    double wts;

    while (!feof(fptr)){
        fgets(a1, sizeof(a1), fptr);
        sscanf(a1, "Atom_type: %d weight: %lf", &elem_type, &wts);


        for (int i=0; i < size_X2; i++){
            fscanf(fptr,"%lf", &X2_train_model[elem_type][count[elem_type]][i]);
        }

        fscanf(fptr, "%*[^\n]\n");
        for (int i=0; i < size_X3; i++){
            fscanf(fptr,"%lf", &X3_train_model[elem_type][count[elem_type]][i]);
        }

        fscanf(fptr, "%*[^\n]\n");
        count[elem_type] = count[elem_type] +1;
   }
   fclose(fptr);


}

