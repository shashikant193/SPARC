#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include "pressure.h"
#include "relax.h"
#include "electrostatics.h"
#include "eigenSolver.h" // Mesh2ChebDegree
#include "tools.h"
#include "mkl.h"
#include "md.h"
#include "isddft.h"
#include "orbitalElecDensInit.h"
#include "initialization.h"
#include "electronicGroundState.h"
#include "stress.h"
#include "tools.h"
#include "sparc_interface.h"
#include "linearsys.h"
#include "regression.h"
#include "tools_mlff.h"
#include "MLFF_read_write.h"
#include "ddbp_tools.h"
#include "ofdft.h"
#include "forces.h"
#include "parallelization.h"
#include "finalization.h"
#include "soap_descriptor.h"
#include "MLFF_merge.h"
#include "tools_mlff.h"
#include "mlff_types.h"

#define max(a,b) ((a)>(b)?(a):(b))

void merge_mlff_atom_data(SPARC_OBJ *pSPARC) {

	
	MLFF_Obj mlff_str;
	char temp_strfname[L_STRING];
	char temp_atomfname[L_STRING];
	char temp_restartfname[L_STRING];
	char a1[512], str[512];
	int check1=0, check2=0, check3=0, check4 = 0;
	int size_X2, size_X3;

	if (pSPARC->descriptor_typ_MLFF==0){
		size_X2 = pSPARC->Ntypes * pSPARC->N_max_SOAP;
		size_X3 = ((pSPARC->Ntypes * pSPARC->N_max_SOAP+1)*(pSPARC->Ntypes * pSPARC->N_max_SOAP))/2 * (pSPARC->L_max_SOAP+1);
	}
	if (pSPARC->descriptor_typ_MLFF==1){
		size_X2 = pSPARC->N_max_SOAP;
		size_X3 = ((pSPARC->N_max_SOAP+1)*(pSPARC->N_max_SOAP))/2 * (pSPARC->L_max_SOAP+1);
	}

	get_N_r_hnl(pSPARC);
	int N_r = pSPARC->N_rgrid_MLFF;

	
	

	mlff_str.N_rgrid = pSPARC->N_rgrid_MLFF;
	mlff_str.n_str_max = pSPARC->n_str_max_mlff;
	mlff_str.n_train_max = pSPARC->n_train_max_mlff;
	mlff_str.size_X2 = size_X2;
	mlff_str.size_X3 = size_X3;
	mlff_str.beta_2 = pSPARC->beta_2_SOAP;
	mlff_str.beta_3 = pSPARC->beta_3_SOAP;
	mlff_str.xi_3 = pSPARC->xi_3_SOAP;
	mlff_str.Nmax = pSPARC->N_max_SOAP;
	mlff_str.Lmax = pSPARC->L_max_SOAP;
	mlff_str.rcut = pSPARC->rcut_SOAP;
	mlff_str.F_tol = pSPARC->F_tol_SOAP;
	mlff_str.sigma_w = 100;
	mlff_str.sigma_v = 0.1;
	mlff_str.kernel_typ = pSPARC->kernel_typ_MLFF;
	mlff_str.rgrid = (double *) malloc(sizeof(double)* N_r);
 	mlff_str.h_nl = (double *) malloc(sizeof(double)* N_r* pSPARC->N_max_SOAP*(pSPARC->L_max_SOAP+1));
	mlff_str.dh_nl = (double *) malloc(sizeof(double)* N_r * pSPARC->N_max_SOAP*(pSPARC->L_max_SOAP+1));
	read_h_nl(pSPARC->N_max_SOAP, pSPARC->L_max_SOAP, mlff_str.rgrid, mlff_str.h_nl, mlff_str.dh_nl, pSPARC);
	mlff_str.cov_train = (double *) malloc(1*sizeof(double));

	int natom_ref[pSPARC->n_merge_mlff][pSPARC->Ntypes], nstr_ref[pSPARC->n_merge_mlff], n_rows[pSPARC->n_merge_mlff];
	double scale_force[pSPARC->n_merge_mlff];

	
	FILE *fptr;

	for (int i = 0; i < pSPARC->n_merge_mlff; i++) {
		// snprintf(temp_strfname, L_STRING, "%s_structures_%02d", pSPARC->fname_merge_prefix);
		// snprintf(temp_atomfname, L_STRING, "%s_atoms_%02d", pSPARC->fname_merge_prefix);
		snprintf(temp_restartfname, L_STRING, "%s_RESTART_%02d.txt", pSPARC->fname_merge_prefix, i+1);
		fptr = fopen(temp_restartfname, "r");
		while (!feof(fptr)){
			fscanf(fptr,"%s",str);
			fscanf(fptr, "%*[^\n]\n");
			if (str[0] == '#' || str[0] == '\n'|| strcmpi(str,"undefined") == 0) {
				fscanf(fptr, "%*[^\n]\n"); // skip current line
				continue;
			}
			if (strcmpi(str,"N_ref_atoms_elemwise:") == 0){
				check1=1;
				for (int j = 0; j < pSPARC->Ntypes; j++){
					fscanf(fptr,"%d", &natom_ref[i][j]);
					// printf("i: %d, j: %d, natom_ref[i][j]: %d\n",i,j,natom_ref[i][j]);
					fscanf(fptr, "%*[^\n]\n");
				}
			} else if (strcmpi(str,"N_ref_str:") == 0) {
				check2=1;
				fscanf(fptr,"%d", &nstr_ref[i]);
				fscanf(fptr, "%*[^\n]\n");
			} else if (strcmpi(str,"SCALE_FORCE_MLFF:") == 0){
				check3=1;
				fscanf(fptr,"%lf", &scale_force[i]);
				fscanf(fptr, "%*[^\n]\n");
			} else if (strcmpi(str,"n_rows:") == 0){
				check4=1;
				fscanf(fptr,"%d", &n_rows[i]);
				fscanf(fptr, "%*[^\n]\n");
			} 
		}
		fclose(fptr);
	}

	int total_str = 0;
	for (int i=0; i<pSPARC->n_merge_mlff; i++ ){
		total_str += nstr_ref[i];
	}
	mlff_str.E_row_idx = (int *)malloc(sizeof(int)*total_str);
	mlff_str.E_row_idx[0]=0;

	mlff_str.nelem = pSPARC->Ntypes;
	mlff_str.natm_train_total = 0;
	mlff_str.natm_train_elemwise = (int *) malloc(sizeof(int)*pSPARC->Ntypes);

	for (int i=0; i < pSPARC->Ntypes; i++){
		mlff_str.natm_train_elemwise[i] = 0;
		for (int j=0; j < pSPARC->n_merge_mlff; j++){
			mlff_str.natm_train_elemwise[i] += natom_ref[j][i];
			mlff_str.natm_train_total += natom_ref[j][i];
		}
	}

	mlff_str.natm_typ_train = (int *) malloc(sizeof(int)*mlff_str.natm_train_total);

	

	int K_size_row = 0;
	int b_size = 0;
	for (int i=0; i<pSPARC->n_merge_mlff; i++){
		K_size_row += n_rows[i];
		b_size += n_rows[i];
		// printf(" n_rows[%d]: %d\n", i,n_rows[i]);
	}
	int K_size_column = mlff_str.natm_train_total;
	
	int w_size = mlff_str.natm_train_total;


	mlff_str.X2_traindataset = (double **) malloc(sizeof(double*)*mlff_str.natm_train_total);
	mlff_str.X3_traindataset = (double **) malloc(sizeof(double*)*mlff_str.natm_train_total);

	for (int i = 0; i < mlff_str.natm_train_total; i++){
		mlff_str.X2_traindataset[i] = (double *) malloc(sizeof(double)*size_X2);
		mlff_str.X3_traindataset[i] = (double *) malloc(sizeof(double)*size_X3);
	}

	mlff_str.K_train = (double **) malloc(sizeof(double*)*K_size_row);
	for (int i =0; i < K_size_row; i++){
		mlff_str.K_train[i] = (double *) malloc(sizeof(double)*K_size_column);
		for (int j=0; j <K_size_column; j++){
			mlff_str.K_train[i][j]=0.0;
		}
	}

	mlff_str.b_no_norm = (double *) malloc(sizeof(double)*b_size);

	mlff_str.weights = (double *) malloc(sizeof(double)*w_size);

	for (int i=0; i < b_size; i++){
		mlff_str.b_no_norm[i] =0.0;
	}

	for (int i=0; i < w_size; i++){
		mlff_str.weights[i] =0.0;
	}




	mlff_str.n_str = 0;
	mlff_str.n_rows = 0;
	mlff_str.n_cols = mlff_str.natm_train_total;

	int temp;
	int count=0, count1;
	for (int i = 0; i < pSPARC->n_merge_mlff; i++){
		snprintf(temp_atomfname, L_STRING, "%s_atoms_%02d.txt", pSPARC->fname_merge_prefix, i+1);
		printf("%s\n",temp_atomfname);
		fptr = fopen(temp_atomfname, "r");	
		fscanf(fptr,"%s",str);
		fscanf(fptr,"%d",&temp);
		if (temp != pSPARC->Ntypes){
			printf("Number of element types don't match in %s %d\n",temp_atomfname, temp);
			exit(1);
		}
		for (int j=0; j<pSPARC->Ntypes; j++){
			fscanf(fptr,"%s",str);
			fscanf(fptr,"%s",str);
			fscanf(fptr,"%s",str);
			fscanf(fptr,"%d",&temp);
			if (temp != natom_ref[i][j]){
				printf("Number of atoms in the training dataset element wise doesn't match between restart and atom ref files\n");
				exit(1);
			}
		}
		for (int j=0; j < pSPARC->Ntypes; j++){
			fscanf(fptr,"%s",str);
			fscanf(fptr,"%s",str);
			fscanf(fptr,"%s",str);
			fscanf(fptr,"%s",str);
			fscanf(fptr,"%s",str);
			for (int k=0; k < natom_ref[i][j]; k++){
				fscanf(fptr,"%s",str);
				fscanf(fptr,"%s",str);
				fscanf(fptr,"%s",str);
				fscanf(fptr,"%s",str);
				fscanf(fptr,"%s",str);
				for (int l=0; l < size_X2; l++)
					fscanf(fptr,"%lf",&mlff_str.X2_traindataset[count][l]);

				for (int l=0; l < size_X3; l++)
					fscanf(fptr,"%lf",&mlff_str.X3_traindataset[count][l]);
				mlff_str.natm_typ_train[count] = j;
				// printf("count: %d, mlff_str.natm_typ_train[count]: %d\n",count,mlff_str.natm_typ_train[count]);
				
				count++;
			}
		}	

		fclose(fptr);
	}

	printf("mlff_str.natm_typ_train[0]: %d\n",mlff_str.natm_typ_train[0]);

	printf("Read all the atoms files\n");

	double CELL[3];
	int natom;
	int natom_elem[pSPARC->Ntypes];
	double *atom_pos, *forces;
	double E;

	int BC[3];

	BC[0] = 1; BC[1] = 1; BC[2] = 1;

	int count_str = 0;
	for (int i = 0; i < pSPARC->n_merge_mlff; i++){
		snprintf(temp_strfname, L_STRING, "%s_structures_%02d.txt", pSPARC->fname_merge_prefix, i+1);
		fptr = fopen(temp_strfname, "r");
		fscanf(fptr,"%s",str);
		fscanf(fptr,"%s",str);
		fscanf(fptr,"%s",str);
		fscanf(fptr,"%s",str);
		fscanf(fptr,"%s",str);


		for (int j=0; j < nstr_ref[i]; j++){
			fscanf(fptr,"%s",str);
			fscanf(fptr,"%d",&temp);
			if (temp != j+1){
				printf("Issue in numbering of structure_no in the file %s\n",temp_strfname);
				exit(1);
			}
			fscanf(fptr,"%s",str);
			fscanf(fptr,"%lf",&CELL[0]);
			fscanf(fptr,"%lf",&CELL[1]);
			fscanf(fptr,"%lf",&CELL[2]);

			fscanf(fptr,"%s",str);
			fscanf(fptr,"%d",&natom);
			mlff_str.E_row_idx[count_str+1] = mlff_str.E_row_idx[count_str]+1+3*natom;

			

			atom_pos = (double *) malloc(sizeof(double)*natom*3);
			forces = (double *) malloc(sizeof(double)*natom*3);

			fscanf(fptr,"%s",str);
			for (int k=0; k<pSPARC->Ntypes; k++){
				fscanf(fptr,"%d",&natom_elem);
			}

			fscanf(fptr,"%s",str);
			for (int k=0; k<3*natom; k++){
				fscanf(fptr,"%lf",&atom_pos[k]);
			}

			fscanf(fptr,"%s",str);
			fscanf(fptr,"%lf",&E);

			fscanf(fptr,"%s",str);
			for (int k=0; k<3*natom; k++){
				fscanf(fptr,"%lf",&forces[k]);
				forces[k] = -1.0*forces[k];    // negative grad(E)
			}
			int *atomtyp;
			atomtyp = (int *) malloc(pSPARC->n_atom*sizeof(int));
			count1 = 0;
			for (int i=0; i < pSPARC->Ntypes; i++){
				for (int j=0; j < natom_elem[i]; j++){
					atomtyp[count1] = i;
					count1++;
				}
			}

			NeighList nlist;
			SoapObj soap_str;



			initialize_nlist(&nlist, natom, mlff_str.rcut, pSPARC->Ntypes);
			build_nlist(mlff_str.rcut, pSPARC->Ntypes, natom, atom_pos, atomtyp, BC, CELL, &nlist);
			build_soapObj(&soap_str, &nlist, mlff_str.rgrid, mlff_str.h_nl, mlff_str.dh_nl, atom_pos, mlff_str.Nmax,
						 mlff_str.Lmax, mlff_str.beta_3, mlff_str.xi_3, pSPARC->N_rgrid_MLFF);
			add_newstr_rows_merge(&soap_str, &nlist, &mlff_str, E/natom, forces);
			printf("Merged a structures\n");
			free(atomtyp);
			delete_soapObj(&soap_str);
			clear_nlist(&nlist);
			free(atom_pos);
			free(forces);
			count_str++;
		}
		fclose(fptr);
	}


	mlff_train_Bayesian_merge(&mlff_str);
	// FILE *fp;
	// fp = fopen("Ktrain_new.txt","w");
	// for (int i=0; i <mlff_str.n_rows; i++){
	// 	for (int j=0; j < mlff_str.n_cols; j++){
	// 		fprintf(fp,"%f ",mlff_str.K_train[i][j]);
	// 	}
	// 	fprintf(fp,"\n");
	// }
	// fclose(fp);
}


void merge_mlff_only_str_data(SPARC_OBJ *pSPARC) {

	
	MLFF_Obj mlff_str;
	char temp_strfname[L_STRING];
	char temp_atomfname[L_STRING];
	char temp_restartfname[L_STRING];
	char a1[512], str[512];
	int check1=0, check2=0, check3=0, check4 = 0;
	int size_X2, size_X3;

	if (pSPARC->descriptor_typ_MLFF==0){
		size_X2 = pSPARC->Ntypes * pSPARC->N_max_SOAP;
		size_X3 = ((pSPARC->Ntypes * pSPARC->N_max_SOAP+1)*(pSPARC->Ntypes * pSPARC->N_max_SOAP))/2 * (pSPARC->L_max_SOAP+1);
	}
	if (pSPARC->descriptor_typ_MLFF==1){
		size_X2 = pSPARC->N_max_SOAP;
		size_X3 = ((pSPARC->N_max_SOAP+1)*(pSPARC->N_max_SOAP))/2 * (pSPARC->L_max_SOAP+1);
	}

	get_N_r_hnl(pSPARC);
	int N_r = pSPARC->N_rgrid_MLFF;
	mlff_str.nelem = 1;
	mlff_str.N_rgrid = pSPARC->N_rgrid_MLFF;
	mlff_str.n_str_max = pSPARC->n_str_max_mlff;
	mlff_str.n_train_max = pSPARC->n_train_max_mlff;
	mlff_str.size_X2 = size_X2;
	mlff_str.size_X3 = size_X3;
	mlff_str.beta_2 = pSPARC->beta_2_SOAP;
	mlff_str.beta_3 = pSPARC->beta_3_SOAP;
	mlff_str.xi_3 = pSPARC->xi_3_SOAP;
	mlff_str.Nmax = pSPARC->N_max_SOAP;
	mlff_str.Lmax = pSPARC->L_max_SOAP;
	mlff_str.rcut = pSPARC->rcut_SOAP;
	mlff_str.F_tol = pSPARC->F_tol_SOAP;
	mlff_str.sigma_w = 100;
	mlff_str.sigma_v = 0.1;
	mlff_str.kernel_typ = pSPARC->kernel_typ_MLFF;
	mlff_str.rgrid = (double *) malloc(sizeof(double)* N_r);
 	mlff_str.h_nl = (double *) malloc(sizeof(double)* N_r* pSPARC->N_max_SOAP*(pSPARC->L_max_SOAP+1));
	mlff_str.dh_nl = (double *) malloc(sizeof(double)* N_r * pSPARC->N_max_SOAP*(pSPARC->L_max_SOAP+1));
	read_h_nl(pSPARC->N_max_SOAP, pSPARC->L_max_SOAP, mlff_str.rgrid, mlff_str.h_nl, mlff_str.dh_nl, pSPARC);
	mlff_str.cov_train = (double *) malloc(1*sizeof(double));

	int natom_ref[pSPARC->n_merge_mlff][pSPARC->Ntypes], nstr_ref[pSPARC->n_merge_mlff], n_rows[pSPARC->n_merge_mlff];

	mlff_str.cov_train = (double *) malloc(1*sizeof(double));
	mlff_str.natm_train_elemwise = (int *) malloc(1 * sizeof(int));
	for (int i=0; i<1; i++){
		mlff_str.natm_train_elemwise[i] = 0;
	}
	mlff_str.natm_typ_train = (int *)malloc(sizeof(int)*1 * pSPARC->n_train_max_mlff);


	int K_size_row = pSPARC->n_str_max_mlff*10;
	int K_size_column = pSPARC->n_train_max_mlff*10;
	int b_size = pSPARC->n_str_max_mlff*100;
	int w_size =10 * pSPARC->n_train_max_mlff;

	mlff_str.K_train = (double **) malloc(sizeof(double*)*K_size_row);
	for (int i =0; i < K_size_row; i++){
		mlff_str.K_train[i] = (double *) malloc(sizeof(double)*K_size_column);
	}

	mlff_str.b_no_norm = (double *) malloc(sizeof(double)*b_size);
	mlff_str.weights = (double *) malloc(sizeof(double)*w_size);

	for (int i = 0; i < K_size_row; i++)
		for (int j=0; j < K_size_column; j++)
			mlff_str.K_train[i][j] = 0;
	
	for (int i = 0; i < b_size; i++)
		mlff_str.b_no_norm[i] = 0;

	for (int i = 0; i < w_size; i++)
		mlff_str.weights[i] = 0;

	mlff_str.soap_descriptor_strdataset = (SoapObj *) malloc(sizeof(SoapObj)*pSPARC->n_str_max_mlff*10);
	mlff_str.X2_traindataset = (double **) malloc(sizeof(double*)*pSPARC->n_train_max_mlff*10);
	mlff_str.X3_traindataset = (double **) malloc(sizeof(double*)*pSPARC->n_train_max_mlff*10);
	for (int i = 0; i < pSPARC->n_train_max_mlff*10; i++){
		mlff_str.X2_traindataset[i] = (double *) malloc(sizeof(double)*size_X2);
		mlff_str.X3_traindataset[i] = (double *) malloc(sizeof(double)*size_X3);
	}



	FILE *fptr, *fptr1;
	double CELL[3];
	int natom;
	int natom_elem[pSPARC->Ntypes];
	double *atom_pos, *forces;
	double E;int temp;

	fptr1 = fopen("nstr_data.txt","r");
	for (int i=0; i < pSPARC->n_merge_mlff; i++){
		fscanf(fptr1,"%d",&nstr_ref[i]);
	}
	fclose(fptr1);

	int total_str = 0;
	for (int i=0; i<pSPARC->n_merge_mlff; i++ ){
		total_str += nstr_ref[i];
	}
	mlff_str.E_row_idx = (int *)malloc(sizeof(int)*total_str);
	mlff_str.E_row_idx[0]=0;


	int BC[3];

	BC[0] = 1; BC[1] = 1; BC[2] = 1;
	int count1;
	int count_str=0;
	double *X2_add_cols, *X3_add_cols;

	X2_add_cols =(double *)malloc(sizeof(double)*size_X2);
	X3_add_cols =(double *)malloc(sizeof(double)*size_X3);

	for (int i = 0; i < pSPARC->n_merge_mlff; i++){
		snprintf(temp_strfname, L_STRING, "%s_structures_strain_%02d.txt", pSPARC->fname_merge_prefix, i+1);
		fptr = fopen(temp_strfname, "r");
		fscanf(fptr,"%s",str);
		fscanf(fptr,"%s",str);
		fscanf(fptr,"%s",str);
		fscanf(fptr,"%s",str);
		fscanf(fptr,"%s",str);

		for (int j=0; j < nstr_ref[i]; j++){
			fscanf(fptr,"%s",str);
			fscanf(fptr,"%d",&temp);
			if (temp != j){
				printf("Issue in numbering of structure_no in the file %s (tmep: %d, j: %d), i: %d, nstr_ref[i]: %d\n",
				temp_strfname, temp, j, i, nstr_ref[i]);
				exit(1);
			}

			fscanf(fptr,"%s",str);
			fscanf(fptr,"%lf",&CELL[0]);
			fscanf(fptr,"%lf",&CELL[1]);
			fscanf(fptr,"%lf",&CELL[2]);
			

			fscanf(fptr,"%s",str);
			fscanf(fptr,"%d",&natom);

			mlff_str.E_row_idx[count_str+1] = mlff_str.E_row_idx[count_str]+1+3*natom;


			atom_pos = (double *) malloc(sizeof(double)*natom*3);
			forces = (double *) malloc(sizeof(double)*natom*3);

			fscanf(fptr,"%s",str);
			for (int k=0; k<pSPARC->Ntypes; k++){
				fscanf(fptr,"%d",&natom_elem[k]);
			}

			fscanf(fptr,"%s",str);
			for (int k=0; k<3*natom; k++){
				fscanf(fptr,"%lf",&atom_pos[k]);
			}

			fscanf(fptr,"%s",str);
			fscanf(fptr,"%lf",&E);
			E = -1*E;



			fscanf(fptr,"%s",str);
			for (int k=0; k<3*natom; k++){
				fscanf(fptr,"%lf",&forces[k]);
				forces[k] = -1.0*forces[k];    // negative grad(E)

			}

			int *atomtyp;
			atomtyp = (int *) malloc(pSPARC->n_atom*sizeof(int));
			count1 = 0;
			for (int ii=0; ii < pSPARC->Ntypes; ii++){
				for (int j=0; j < natom_elem[ii]; j++){
					atomtyp[count1] = ii;
					count1++;
				}
			}
			NeighList nlist;
			SoapObj soap_str;
			initialize_nlist(&nlist, natom, mlff_str.rcut, pSPARC->Ntypes);
			build_nlist(mlff_str.rcut, pSPARC->Ntypes, natom, atom_pos, atomtyp, BC, CELL, &nlist);
			build_soapObj(&soap_str, &nlist, mlff_str.rgrid, mlff_str.h_nl, mlff_str.dh_nl, atom_pos, mlff_str.Nmax,
						 mlff_str.Lmax, mlff_str.beta_3, mlff_str.xi_3, pSPARC->N_rgrid_MLFF);
			for (int des_arr = 0; des_arr < size_X2; des_arr++){
				X2_add_cols[des_arr] = soap_str.X2[0][des_arr];
			}
			for (int des_arr = 0; des_arr < size_X3; des_arr++){
				X3_add_cols[des_arr] = soap_str.X3[0][des_arr];
			}
			if (count_str==0){
				add_firstMD(&soap_str, &nlist, &mlff_str, E/natom, forces);
			} else {
				add_newstr_rows(&soap_str, &nlist, &mlff_str, E/natom, forces);
				add_newtrain_cols(X2_add_cols, X3_add_cols, 0, &mlff_str);
			}

			count_str += 1;
			free(atomtyp);
			free(atom_pos);
			free(forces);

		}
		fclose(fptr);
	}
	mlff_train_Bayesian_merge(&mlff_str);


}



void add_newstr_rows_merge(SoapObj *soap_str, NeighList *nlist, MLFF_Obj *mlff_str, double E, double *F) {
	int row_idx, col_idx, atom_idx, natom = soap_str->natom, nelem = soap_str->nelem, num_Fterms_exist, num_Fterms_newstr, i, iel, iatm_el_tr, iatm_str, neighs, istress;
	int size_X2 = soap_str->size_X2, size_X3 = soap_str->size_X3;
	double beta_2 = soap_str->beta_2, beta_3 = soap_str->beta_3, xi_3 = soap_str->xi_3;
	double old_mu_stress, old_std_stress, old_mu_E, old_std_E, old_std_F, newstr_std_F;

	int kernel_typ = mlff_str->kernel_typ;

	old_mu_E = mlff_str->mu_E;
	old_std_E = mlff_str->std_E;
	old_std_F = mlff_str->std_F;
	mlff_str->mu_E  = old_mu_E *(mlff_str->n_str/(mlff_str->n_str+1.0)) + E/(mlff_str->n_str+1.0);

	// printf("mlff_str->n_str: %d",mlff_str->n_str);
	if (mlff_str->n_str==0) {
		mlff_str->mu_E = E;
		mlff_str->std_E  = 1;
		mlff_str->std_F  = sqrt(get_variance(F, 3*natom));
	} else if (mlff_str->n_str==1){
		double E_mean = 0.5*(old_mu_E+E);
		mlff_str->std_E = sqrt(((E-E_mean)*(E-E_mean) +(old_mu_E-E_mean)*(old_mu_E-E_mean))/mlff_str->n_str);

	} else if (mlff_str->n_str==2) {
		double E0 = mlff_str->b_no_norm[0], E1 = mlff_str->b_no_norm[7+3*natom], E2 = E;
		double E_mean = (1.0/3.0)*(E0+E1+E2);
		mlff_str->std_E = sqrt((E0-E_mean)*(E0-E_mean)+(E1-E_mean)*(E1-E_mean)+(E2-E_mean)*(E2-E_mean)/2.0);
	}else {

		mlff_str->std_E = sqrt(((mlff_str->n_str-1.0)*old_std_E*old_std_E + (E-mlff_str->mu_E)*(E-old_mu_E))/mlff_str->n_str);
	}
	if (mlff_str->n_str > 0){
		newstr_std_F = sqrt(get_variance(F, 3*natom));
		num_Fterms_exist = mlff_str->n_rows - mlff_str->n_str*7;
		num_Fterms_newstr = 3*soap_str->natom;

		mlff_str->std_F = sqrt((double) (num_Fterms_exist-1)/(double) (num_Fterms_exist+num_Fterms_newstr-1.0) *(old_std_F*old_std_F)
							+ (double) (num_Fterms_newstr-1)/(double) (num_Fterms_exist+num_Fterms_newstr-1.0) * (newstr_std_F*newstr_std_F));

	}

	
	// for (i=0; i<6; i++){
	// 	old_mu_stress = mlff_str->mu_stress[i];
	// 	old_std_stress = mlff_str->std_stress[i];
	// 	mlff_str->mu_stress[i]  = old_mu_stress*(mlff_str->n_str/(mlff_str->n_str+1.0)) + au2GPa*stress[i]/(double) (mlff_str->n_str+1.0);
	// 	if (mlff_str->n_str==1){
	// 		double mean_stress = 0.5*(au2GPa*stress[i]+old_mu_stress);
	// 		mlff_str->std_stress[i] = sqrt(((au2GPa*stress[i]-mean_stress)*(au2GPa*stress[i]-mean_stress) + (old_mu_stress-mean_stress)*(old_mu_stress-mean_stress))/(double)mlff_str->n_str);
	// 	} else if (mlff_str->n_str==2){
	// 		double st0 = mlff_str->b_no_norm[1+3*natom+i], st1 = mlff_str->b_no_norm[7+3*natom+1+3*natom+i], st2 =au2GPa*stress[i];
	// 		double mean_stress = (1.0/3.0)*(st0+st1+st2);
	// 		mlff_str->std_stress[i] = sqrt((st0-mean_stress)*(st0-mean_stress)+(st1-mean_stress)*(st1-mean_stress)+(st2-mean_stress)*(st2-mean_stress)/2.0);
	// 	}else {
	// 		mlff_str->std_stress[i] = sqrt(((double) (mlff_str->n_str-1.0)*old_std_stress*old_std_stress 
	// 			+ (au2GPa*stress[i]-mlff_str->mu_stress[i])*(au2GPa*stress[i]-old_mu_stress))/(double) mlff_str->n_str);
	// 	}
	// }

	printf("std_E: %f, std_F: %f\n",mlff_str->std_E,mlff_str->std_F);
	// printf("std_stress: %f,%f,%f,%f,%f,%f\n",mlff_str->std_stress[0],mlff_str->std_stress[1],mlff_str->std_stress[2],
	// 	mlff_str->std_stress[3],mlff_str->std_stress[4],mlff_str->std_stress[5]);
	// populate bvec (not normalized)
	mlff_str->b_no_norm[mlff_str->n_rows] = E;
	for (i = 0; i < natom; i++){
		mlff_str->b_no_norm[mlff_str->n_rows+3*i+1] = F[3*i];
		mlff_str->b_no_norm[mlff_str->n_rows+3*i+2] = F[3*i+1];
		mlff_str->b_no_norm[mlff_str->n_rows+3*i+3] = F[3*i+2];
	}
	// for (i = 0; i < 6; i++){
	// 	mlff_str->b_no_norm[1 + mlff_str->n_rows + 3*natom + i] = au2GPa*stress[i];
	// }



	int *cum_natm_elem;
	cum_natm_elem = (int *)malloc(sizeof(int)*nelem);
	cum_natm_elem[0] = 0;
	for (int i = 1; i < nelem; i++){
		cum_natm_elem[i] += soap_str->natom_elem[i-1];
	}

	int **cum_natm_ele_cols;
	cum_natm_ele_cols = (int **)malloc(sizeof(int*)*nelem);
	for (int i = 0; i <nelem; i++){
		cum_natm_ele_cols[i] = (int *)malloc(sizeof(int)*mlff_str->natm_train_elemwise[i]);
	}

	// for (int j=0; j < mlff_str->natm_train_total; j++)
	// 	mlff_str->natm_typ_train[j] = 0;

	for (int i = 0; i < nelem; i++){
		int count=0;
		for (int j=0; j < mlff_str->natm_train_total; j++){
			if (mlff_str->natm_typ_train[j] == i){
				cum_natm_ele_cols[i][count] = j;
				// printf("count: %d, i: %d, j: %d, mlff_str->natm_typ_train[j]: %d,cum_natm_ele_cols[i][count]: %d\n",count,i,j,mlff_str->natm_typ_train[j],cum_natm_ele_cols[i][count]);
				count++;
			}
		}
	}


	// Energy term to populate K_train matrix
	double temp = (1.0/ (double) natom);
	for (iel = 0; iel < nelem; iel++){
		for (iatm_el_tr = 0; iatm_el_tr < mlff_str->natm_train_elemwise[iel]; iatm_el_tr++){
			// col_idx = cum_natm_elem[iel] + iatm_el_tr;
			col_idx = cum_natm_ele_cols[iel][iatm_el_tr];
			// if (iel > 0)
			// 	col_idx = mlff_str->natm_train_elemwise[iel-1] + iatm_el_tr;
			for (iatm_str = 0; iatm_str < soap_str->natom_elem[iel]; iatm_str++){
				atom_idx = cum_natm_elem[iel] + iatm_str;
				// if (iel > 0)
				// 	atom_idx = soap_str->natom_elem[iel-1] + iatm_str;
				// printf("came here xx1 col_idx: %d, atom_idx: %d, iel: %d, iatm_el_tr: %d\n", col_idx, atom_idx, iel,iatm_el_tr );
				mlff_str->K_train[mlff_str->n_rows][col_idx] +=  temp* soap_kernel(kernel_typ, soap_str->X2[atom_idx], soap_str->X3[atom_idx],
															mlff_str->X2_traindataset[col_idx], mlff_str->X3_traindataset[col_idx],
												 			beta_2, beta_3, xi_3, size_X2, size_X3);
				// printf("came here xx2\n");
				if (mlff_str->K_train[mlff_str->n_rows][col_idx] > 1.0){
					printf("In add_rows energy term > 1 error, row: %d, col: %d, val: %f\n",
					mlff_str->n_rows,col_idx,mlff_str->K_train[mlff_str->n_rows][col_idx]);
					exit(1);
				}
			}
		}
	}

	// printf("Energy terms in K_train for add_rows\n");
	// for (int i=0; i <mlff_str->n_cols; i++){
	// 	printf("%f ",mlff_str->K_train[mlff_str->n_rows][i]);
	// }
	// printf("\n");

	// printf("ktrain for add rows\n");
	// for (int ii=0; ii<mlff_str->n_cols; ii++){
	// 	printf("%f ",mlff_str->K_train[0][ii]);
	// }
	// printf("\nktrain for add rows end\n");

	// Force terms to populate K_train matrix

	for (iel = 0; iel < nelem; iel++){
		for (iatm_el_tr = 0; iatm_el_tr < mlff_str->natm_train_elemwise[iel]; iatm_el_tr++){
			// col_idx = cum_natm_elem[iel] + iatm_el_tr;
			col_idx = cum_natm_ele_cols[iel][iatm_el_tr];
			
			// if (iel > 0)
			// 	col_idx = mlff_str->natm_train_elemwise[iel-1] + iatm_el_tr;
			for (iatm_str = 0; iatm_str < soap_str->natom_elem[iel]; iatm_str++){
				atom_idx = cum_natm_elem[iel] + iatm_str;
				
				// if (iel>0)
				// 	atom_idx = soap_str->natom_elem[iel-1] + iatm_str;

				row_idx = mlff_str->n_rows + 3*atom_idx+1;
				
				// x-component (w.r.t itself) "because an atom is not considered it's neighbour hence dealt outside the neighs loop"
				mlff_str->K_train[row_idx][col_idx] +=  
					der_soap_kernel(kernel_typ, soap_str->dX2_dX[atom_idx][0], soap_str->dX3_dX[atom_idx][0],
					 soap_str->X2[atom_idx], soap_str->X3[atom_idx],
					mlff_str->X2_traindataset[col_idx], mlff_str->X3_traindataset[col_idx],
					beta_2, beta_3, xi_3, size_X2, size_X3);
				

				// y-component (w.r.t itself) "because an atom is not considered it's neighbour hence dealt outside the neighs loop"
				mlff_str->K_train[row_idx+1][col_idx] +=  
					der_soap_kernel(kernel_typ, soap_str->dX2_dY[atom_idx][0], soap_str->dX3_dY[atom_idx][0],
					 soap_str->X2[atom_idx], soap_str->X3[atom_idx],
					mlff_str->X2_traindataset[col_idx], mlff_str->X3_traindataset[col_idx],
					beta_2, beta_3, xi_3, size_X2, size_X3);


				// z-component (w.r.t itself) "because an atom is not considered it's neighbour hence dealt outside the neighs loop"
				mlff_str->K_train[row_idx+2][col_idx] +=  
					der_soap_kernel(kernel_typ, soap_str->dX2_dZ[atom_idx][0], soap_str->dX3_dZ[atom_idx][0],
					 soap_str->X2[atom_idx], soap_str->X3[atom_idx],
					mlff_str->X3_traindataset[col_idx], mlff_str->X3_traindataset[col_idx],
					beta_2, beta_3, xi_3, size_X2, size_X3);
				for (neighs =0; neighs < soap_str->unique_Nneighbors[atom_idx]; neighs++){
					row_idx = mlff_str->n_rows + 3*soap_str->unique_neighborList[atom_idx].array[neighs] + 1; // Possible source of error
					// x-component (w.r.t neighs neighbour)
					mlff_str->K_train[row_idx][col_idx] +=  
					der_soap_kernel(kernel_typ, soap_str->dX2_dX[atom_idx][1+neighs], soap_str->dX3_dX[atom_idx][1+neighs],
						 soap_str->X2[atom_idx], soap_str->X3[atom_idx],
						mlff_str->X2_traindataset[col_idx], mlff_str->X3_traindataset[col_idx],
						beta_2, beta_3, xi_3, size_X2, size_X3);
					// y-component (w.r.t neighs neighbour)
					mlff_str->K_train[row_idx+1][col_idx] +=  
					der_soap_kernel(kernel_typ, soap_str->dX2_dY[atom_idx][1+neighs], soap_str->dX3_dY[atom_idx][1+neighs], 
						soap_str->X2[atom_idx], soap_str->X3[atom_idx],
						mlff_str->X2_traindataset[col_idx], mlff_str->X3_traindataset[col_idx],
						beta_2, beta_3, xi_3, size_X2, size_X3);
					// z-component (w.r.t neighs neighbour)
					mlff_str->K_train[row_idx+2][col_idx] +=  
					der_soap_kernel(kernel_typ, soap_str->dX2_dZ[atom_idx][1+neighs], soap_str->dX3_dZ[atom_idx][1+neighs], 
						soap_str->X2[atom_idx], soap_str->X3[atom_idx],
						mlff_str->X2_traindataset[col_idx], mlff_str->X3_traindataset[col_idx],
						beta_2, beta_3, xi_3, size_X2, size_X3);
				}

			}
		}
	}


	// Stress terms to populate K_train matrix
	// double volume = soap_str->cell[0] *soap_str->cell[1]*soap_str->cell[2];
	// for (iel = 0; iel < nelem; iel++){
	// 	for (iatm_el_tr = 0; iatm_el_tr < mlff_str->natm_train_elemwise[iel]; iatm_el_tr++){
	// 		// col_idx = cum_natm_elem[iel] + iatm_el_tr;
	// 		col_idx = cum_natm_ele_cols[iel][iatm_el_tr];
	// 		// if (iel > 0)
	// 		// 	col_idx = mlff_str->natm_train_elemwise[iel-1] + iatm_el_tr;
	// 		for (iatm_str = 0; iatm_str < soap_str->natom_elem[iel]; iatm_str++){
	// 			atom_idx = cum_natm_elem[iel] + iatm_str;
	// 			// if (iel>0)
	// 			// 	atom_idx = soap_str->natom_elem[iel-1] + iatm_str;
	// 			for (istress =0; istress < 6; istress++){
	// 				row_idx = mlff_str->n_rows+1+3*soap_str->natom+istress;

	// 				mlff_str->K_train[row_idx][col_idx] +=  au2GPa*(1.0/volume)*
	// 				der_soap_kernel(kernel_typ, soap_str->dX2_dF[atom_idx][istress], soap_str->dX3_dF[atom_idx][istress],
	// 					 soap_str->X2[atom_idx], soap_str->X3[atom_idx],
	// 					mlff_str->X2_traindataset[col_idx], mlff_str->X3_traindataset[col_idx],
	// 					beta_2, beta_3, xi_3, size_X2, size_X3);

	// 			}

	// 		}
	// 	}
	// }
	// updating other MLFF parameters such as number of structures, number of training environment, element typ of training env
	mlff_str->n_str += 1;
	mlff_str->n_rows += 3*soap_str->natom + 1;
	free(cum_natm_elem);
	for (int i = 0; i <nelem; i++){
		free(cum_natm_ele_cols[i]);
	}
	free(cum_natm_ele_cols);
	// free(stress);
}

void mlff_train_Bayesian_merge(MLFF_Obj *mlff_str){

	printf("1 std_F: %f\n",mlff_str->std_F);

    int info, m = mlff_str->n_rows, n = mlff_str->n_cols;
    int i,j,k, count, count1;

    double *a_scaled, *b_scaled;
	printf("2 std_F: %f\n",mlff_str->std_F);
    ///////////////////////////////////////////////////////////////////////////////////
                                //  Scaling applied to design matrices
    ///////////////////////////////////////////////////////////////////////////////////
    mlff_str->E_scale = 1.0;
    mlff_str->F_scale = mlff_str->std_E/mlff_str->std_F;
	mlff_str->relative_scale_F = 0.1;

	printf("3 std_F: %f\n",mlff_str->std_F);
    // mlff_str->stress_scale[0] = mlff_str->std_E/mlff_str->std_stress[0];
    // mlff_str->stress_scale[1] = mlff_str->std_E/mlff_str->std_stress[1];
    // mlff_str->stress_scale[2] = mlff_str->std_E/mlff_str->std_stress[2];
    // mlff_str->stress_scale[3] = mlff_str->std_E/mlff_str->std_stress[3];
    // mlff_str->stress_scale[4] = mlff_str->std_E/mlff_str->std_stress[4];
    // mlff_str->stress_scale[5] = mlff_str->std_E/mlff_str->std_stress[5];


	


    a_scaled = (double *) malloc(m*n * sizeof(double));
    b_scaled = (double *) malloc(m * sizeof(double));

	for (int i=0; i < mlff_str->n_str; i++)
		printf("%d\n",mlff_str->E_row_idx[i]);
	printf("m: %d\n",m);
    double scale =0;
    for (i = 0; i < m; i++ ){
		
        // int quot = i%(1+3*mlff_str->natom);
		int quot = lin_search_INT(mlff_str->E_row_idx, mlff_str->n_str, i);
        if (quot!=-1){
			// printf("i: %d, b_no_norm: %f\n",i, mlff_str->b_no_norm[i]);
            scale = mlff_str->E_scale ;
            b_scaled[i] = (1.0/mlff_str->std_E)*(mlff_str->b_no_norm[i] - mlff_str->mu_E);
        } else {
            scale = mlff_str->F_scale* mlff_str->relative_scale_F;
            b_scaled[i] = (1.0/mlff_str->std_F)*(mlff_str->b_no_norm[i])* mlff_str->relative_scale_F;
        }
        // }else{
        //     scale = mlff_str->stress_scale[quot-3*mlff_str->natom-1]* mlff_str->relative_scale_stress[quot-3*mlff_str->natom-1];
        //     b_scaled[i] = (1.0/mlff_str->std_stress[quot-3*mlff_str->natom-1])*(mlff_str->b_no_norm[i])* mlff_str->relative_scale_stress[quot-3*mlff_str->natom-1];
        // }

        for (j = 0; j < n; j++){
            a_scaled[j*m+i] = scale * mlff_str->K_train[i][j];
        }
    }

	printf("4 std_F: %f\n",mlff_str->std_F);

    double *AtA, *Atb, *AtA_h, *Atb_h;

    AtA = (double *) malloc(sizeof(double)* mlff_str->n_cols * mlff_str->n_cols);
    Atb = (double *) malloc(sizeof(double)* mlff_str->n_cols);

    AtA_h = (double *) malloc(sizeof(double)* mlff_str->n_cols * mlff_str->n_cols);
    Atb_h = (double *) malloc(sizeof(double)* mlff_str->n_cols);

    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
                mlff_str->n_cols, mlff_str->n_cols, mlff_str->n_rows, 1.0, a_scaled, mlff_str->n_rows, a_scaled, mlff_str->n_rows, 0.0, AtA, mlff_str->n_cols);

    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
                mlff_str->n_cols, 1, mlff_str->n_rows, 1.0, a_scaled, mlff_str->n_rows, b_scaled, mlff_str->n_rows, 0.0, Atb, mlff_str->n_cols);
	printf("5 std_F: %f\n",mlff_str->std_F);
   
    // FILE *f1, *f2;
    // f1 = fopen("a_scaled.txt","w");
    // f2 = fopen("b_scaled.txt","w");

    // for (i = 0; i < m; i++){
    //   fprintf(f2,"%10.9f\n",b_scaled[i]);
    //   for (j = 0; j < n; j++){
    //     fprintf(f1,"%10.9f ",a_scaled[j*m+i]);
    //   }
    //   fprintf(f1,"\n");
    // }
    // fclose(f1);
    // fclose(f2);


    for (int i=0; i < mlff_str->n_cols * mlff_str->n_cols; i++){
    	AtA_h[i] = AtA[i];
    }

    for (int i=0; i < mlff_str->n_cols; i++){
    	Atb_h[i] = Atb[i];
    }
	printf("5 std_F: %f\n",mlff_str->std_F);

    int dohyperparameter=1;
    double sigma_w, sigma_v;
    if (dohyperparameter){
    	hyperparameter_Bayesian(a_scaled, AtA_h, Atb_h, b_scaled, mlff_str);
    }

	printf("5-1 std_F: %f\n",mlff_str->std_F);

    for (int i =0; i <mlff_str->n_cols; i++){  
      AtA[i*mlff_str->n_cols+i] += (mlff_str->sigma_v*mlff_str->sigma_v)/(mlff_str->sigma_w*mlff_str->sigma_w);

    }

	printf("6 std_F: %f\n",mlff_str->std_F);

    int ipiv[mlff_str->n_cols];
    // info = LAPACKE_dgesv( LAPACK_COL_MAJOR, mlff_str->n_cols, 1, AtA, mlff_str->n_cols, &ipiv[0],
    //                      Atb, mlff_str->n_cols );

    // for (int i=0; i < mlff_str->n_cols; i++)
    // 	mlff_str->weights[i] = Atb[i];

    free(mlff_str->cov_train);
    mlff_str->cov_train = (double *)malloc(mlff_str->n_cols*mlff_str->n_cols*sizeof(double));
	printf("7 std_F: %f\n",mlff_str->std_F);
    // FILE *f1, *f2;
    // f1 = fopen("AtA.txt","w");
    // f2 = fopen("AtAinv.txt","w");
    // for (int i=0; i < mlff_str->n_cols; i++){
    //   for (int j=0; j <mlff_str->n_cols; j++){
    //     fprintf(f1,"%f ",AtA[j*mlff_str->n_cols + i]);
    //   }
    //   fprintf(f1,"\n");
    // }
    // fclose(f1);
    info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, mlff_str->n_cols, mlff_str->n_cols, AtA, mlff_str->n_cols, &ipiv[0]);
    info = LAPACKE_dgetri(LAPACK_COL_MAJOR, mlff_str->n_cols, AtA, mlff_str->n_cols, &ipiv[0]);
	printf("8 std_F: %f\n",mlff_str->std_F);
    // for (int i=0; i < mlff_str->n_cols; i++){
    //   for (int j=0; j <mlff_str->n_cols; j++){
    //     fprintf(f2,"%f ",AtA[j*mlff_str->n_cols + i]);
    //   }
    //   fprintf(f2,"\n");
    // }
    // fclose(f2);

    // exit(1);

    for (int i = 0; i < mlff_str->n_cols*mlff_str->n_cols; i++){
    	mlff_str->cov_train[i] = AtA[i] * mlff_str->sigma_v*mlff_str->sigma_v;
    }
	printf("9 std_F: %f\n",mlff_str->std_F);
	char str1[512] = "MLFF_data_reference_atoms.txt"; 
    strcpy(mlff_str->ref_atom_name, str1);
	char str2[512] = "MLFF_RESTART.txt"; 
    strcpy(mlff_str->restart_name, str2);

	printf("came here in the end\n");

	printf("std_F: %f\n",mlff_str->std_F);
    print_restart_MLFF(mlff_str);
	printf("wrote restart file\n");
    print_ref_atom_MLFF(mlff_str);
	printf("wrote ref atom file\n");

    free(a_scaled);
    free(b_scaled);
    free(AtA);
    free(Atb);
    free(AtA_h);
    free(Atb_h);
}