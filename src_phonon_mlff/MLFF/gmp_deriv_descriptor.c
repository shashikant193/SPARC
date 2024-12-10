#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <time.h>
#include "helper.h"
#include "solid_harmonics.h"
#include "surface_harmonics.h"
#include "gmp_deriv_descriptor.h"
#include "ddbp_tools.h"
#include "soap_descriptor.h"

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))

/*
initialize_nlist function initializes the a objects in the NeighList structure and also allocates memory for the dunamic arrays.

[Input]
1. natom: Number of atoms in the system
2. rcut: Cutoff distance (bohr)
3. nelem: Number of element species 
[Output]
1. nlist: pointer to Neighlist structure to be initialized
*/
/*

initialize_GMPObj function initializes the objects in GMPObj structure and also allocates memory of dynamic arrays

[Input]
1. nlist: pointer to Neighlist structure
2. cal_atoms: pointer to list of indexes of atoms requiring descriptor calculations
3. cal_num: number of atoms requiring descriptor calculations
4. integer GMP params: mcsh order, square (1) or square root (0), solid (1) or surface (0) harmonic
5. double GMP params: gaussian probe sigma, 1 (unexplained, never referenced - will change later), sigma manipulation, sigma manipulation, sigma manipulation
6. nmcsh: total number of descriptors/atom
7. atom_gaussian: pointer to primitive gaussian parameters
8. ngaussians: pointer to number of primitives
9. element_index_to_order: param to convert atomic number to order in list of gaussian parameters
[Output]
1. gmp_str: pointer to GMPObj structure

*/

void initialize_gmpObj(GMPObj *gmp_str, NeighList *nlist, int *cal_atoms, int cal_num, int **params_i, double **params_d, int nmcsh, double** atom_gaussian, int* ngaussians, int* element_index_to_order){
    int size_X, i,j,k;
    int nelem = nlist->nelem, natom = nlist->natom;
    size_X = nmcsh;
    gmp_str->size_X = size_X;
    gmp_str->cal_atoms = (int *)malloc(sizeof(int)*nlist->natom);
    for (int i = 0; i <nlist->natom; i++){
        gmp_str->cal_atoms[i] = cal_atoms[i];
    }
    // gmp_str->cal_atoms = cal_atoms;

    gmp_str->cal_num = cal_num;

    gmp_str->params_i = (int **) malloc(sizeof(int*)*nmcsh);
    for (int i=0; i < nmcsh; i++){
        gmp_str->params_i[i] = (int *)malloc(sizeof(int)*3);
    }
    for (int i=0; i < nmcsh; i++){
        for (int j=0; j < 3; j++){
            gmp_str->params_i[i][j] = params_i[i][j];
        }
    }
    // gmp_str->params_i = params_i;

    gmp_str->params_d = (double **) malloc(sizeof(double*)*nmcsh);
    for (int i = 0; i < nmcsh; i++) gmp_str->params_d[i] = (double*) malloc(sizeof(double)*6);
    for (int i = 0; i < nmcsh; i++){
        for (int j=0; j < 6; j++){
            gmp_str->params_d[i][j] = params_d[i][j];
        }
    }
    // gmp_str->params_d = params_d;

    gmp_str->nmcsh = nmcsh;

    gmp_str->atom_gaussian = (double **)malloc(nelem*sizeof(double*));
    for (int j=0; j < nelem; j++) gmp_str->atom_gaussian[j] = (double *)malloc(8*sizeof(double));
    for (int i = 0; i < 8; i++){
        for (int j=0; j < nelem; j++){
            gmp_str->atom_gaussian[j][i] = atom_gaussian[j][i];
        }   
    }
    // gmp_str->atom_gaussian = atom_gaussian;

    gmp_str->ngaussians = (int *)malloc(sizeof(int)*nelem);
    for (int j=0; j < nelem; j++) gmp_str->ngaussians[j] = ngaussians[j];
    // gmp_str->ngaussians = ngaussians;

    gmp_str->element_index_to_order = (int *)malloc(sizeof(int)*120);
    for (int i = 0; i < 120; i++){
        gmp_str->element_index_to_order[i] = element_index_to_order[i];
    }
    // gmp_str->element_index_to_order = element_index_to_order;

    gmp_str->natom = natom;
    gmp_str->nelem = nelem;
    gmp_str->rcut = nlist->rcut;
    gmp_str->cell[0] = nlist->cell[0];
    gmp_str->cell[1] = nlist->cell[1];
    gmp_str->cell[2] = nlist->cell[2];

    gmp_str->Nneighbors = (int *) malloc(sizeof(int)*natom);
    for (i=0; i < natom; i++){
        gmp_str->Nneighbors[i] = nlist->Nneighbors[i];

    }
    gmp_str->unique_Nneighbors = (int *) malloc(sizeof(int)*natom);
    for (i=0; i < natom; i ++){
        gmp_str->unique_Nneighbors[i] = nlist->unique_Nneighbors[i];

    }

    gmp_str->unique_Nneighbors_elemWise = (int **) malloc(natom*sizeof(int*));
	for (i=0; i < natom; i++){
		gmp_str->unique_Nneighbors_elemWise[i] = (int *) malloc(nelem*sizeof(int));
	}
	for (i=0; i < natom; i++){
		for(j=0; j < nelem; j++){
			gmp_str->unique_Nneighbors_elemWise[i][j] = nlist->unique_Nneighbors_elemWise[i][j];
		}
	}

	gmp_str->unique_neighborList_elemWise = (dyArray **) malloc(sizeof(dyArray*)*natom);
	for (i =0; i < natom; i++){
		gmp_str->unique_neighborList_elemWise[i] = (dyArray *) malloc(sizeof(dyArray)*nelem);
	}

	for (i =0; i < natom; i++){
		for(j=0; j < nelem; j++){
			gmp_str->unique_neighborList_elemWise[i][j].len = nlist->unique_neighborList_elemWise[i][j].len;
			gmp_str->unique_neighborList_elemWise[i][j].capacity = nlist->unique_neighborList_elemWise[i][j].capacity;
			gmp_str->unique_neighborList_elemWise[i][j].array = (int *)malloc(sizeof(int)*nlist->unique_neighborList_elemWise[i][j].len);
			for (k =0; k < nlist->unique_neighborList_elemWise[i][j].len; k++){
				gmp_str->unique_neighborList_elemWise[i][j].array[k] = nlist->unique_neighborList_elemWise[i][j].array[k];
			}
		}
	}

	gmp_str->neighborList = (dyArray *) malloc(sizeof(dyArray)*natom);
	for (i =0; i < natom; i++){
		gmp_str->neighborList[i].len = nlist->neighborList[i].len;
		gmp_str->neighborList[i].capacity = nlist->neighborList[i].capacity;
		gmp_str->neighborList[i].array = (int *)malloc(sizeof(int)*nlist->neighborList[i].len);
		for (k =0; k<nlist->neighborList[i].len; k++){
			gmp_str->neighborList[i].array[k] = nlist->neighborList[i].array[k];
		}
	}

	gmp_str->unique_neighborList = (dyArray *) malloc(sizeof(dyArray)*natom);
	for (i =0; i < natom; i++){
		gmp_str->unique_neighborList[i].len = nlist->unique_neighborList[i].len;
		gmp_str->unique_neighborList[i].capacity = nlist->unique_neighborList[i].capacity;
		gmp_str->unique_neighborList[i].array = (int *)malloc(sizeof(int)*nlist->unique_neighborList[i].len);
		for (k =0; k<nlist->unique_neighborList[i].len; k++){
			gmp_str->unique_neighborList[i].array[k] = nlist->unique_neighborList[i].array[k];
		}
	}


	gmp_str->natom_elem = (int *) malloc(sizeof(int)*nelem);
	for (i=0; i<nelem; i++){
		gmp_str->natom_elem[i] = nlist->natom_elem[i];
	}

    gmp_str->X = (double **) malloc(natom * sizeof(double*));
    gmp_str->dX_dX = (double ***) malloc(natom * sizeof(double**));
	gmp_str->dX_dY = (double ***) malloc(natom * sizeof(double**));
	gmp_str->dX_dZ = (double ***) malloc(natom * sizeof(double**));

    for (i=0; i < natom; i++){
		gmp_str->X[i] = (double *) malloc(size_X * sizeof(double));
		for (int sz=0; sz < size_X; sz++){
			gmp_str->X[i][sz] = 0.0;
		}

		int uniq_natms = uniqueEle((nlist->neighborList[i]).array, nlist->Nneighbors[i]);
		gmp_str->dX_dX[i] = (double **) malloc((uniq_natms) * sizeof(double*));
		gmp_str->dX_dY[i] = (double **) malloc((uniq_natms) * sizeof(double*));
		gmp_str->dX_dZ[i] = (double **) malloc((uniq_natms) * sizeof(double*));

		for (j=0; j < uniq_natms; j++){
			gmp_str->dX_dX[i][j] = (double *) malloc(size_X * sizeof(double));
			gmp_str->dX_dY[i][j] = (double *) malloc(size_X * sizeof(double));
			gmp_str->dX_dZ[i][j] = (double *) malloc(size_X * sizeof(double));
            for (int sz=0; sz < size_X; sz++){
				gmp_str->dX_dX[i][j][sz] = 0.0;
				gmp_str->dX_dY[i][j][sz] = 0.0;
				gmp_str->dX_dZ[i][j][sz] = 0.0;
			}
        }
    }
}

void free_GMP(GMPObj *gmp_str){
    free(gmp_str->cal_atoms);
    for (int i = 0; i < gmp_str->nmcsh; i++) free(gmp_str->params_d[i]);
    free(gmp_str->params_d);
    for (int i = 0; i < gmp_str->nmcsh; i++) free(gmp_str->params_i[i]);
    free(gmp_str->params_i);
    for (int i=0; i<gmp_str->nelem; i++) free(gmp_str->atom_gaussian[i]);
    free(gmp_str->atom_gaussian);
    free(gmp_str->ngaussians);
    free(gmp_str->element_index_to_order);
    free(gmp_str->unique_Nneighbors);
    free(gmp_str->natom_elem);
    for (int i=0; i<gmp_str->natom; i++){
        free(gmp_str->X[i]);
        free(gmp_str->unique_Nneighbors_elemWise[i]);

        for (int j=0; j < gmp_str->nelem; j++){
            delete_dyarray(&(gmp_str->unique_neighborList_elemWise[i][j]));
        }

        free(gmp_str->unique_neighborList_elemWise[i]);
        // int uniq_natms = uniqueEle((gmp_str->neighborList[i]).array, gmp_str->Nneighbors[i]);
        delete_dyarray(&(gmp_str->neighborList[i]));
        delete_dyarray(&(gmp_str->unique_neighborList[i]));
        for (int j=0; j<gmp_str->natom; j++){
            free(gmp_str->dX_dX[i][j]);
            free(gmp_str->dX_dY[i][j]);
            free(gmp_str->dX_dZ[i][j]);
        }
        free(gmp_str->dX_dX[i]);
        free(gmp_str->dX_dY[i]);
        free(gmp_str->dX_dZ[i]);
    }

    free(gmp_str->Nneighbors);
    free(gmp_str->unique_Nneighbors_elemWise);
    free(gmp_str->unique_neighborList_elemWise);
    free(gmp_str->neighborList);
    free(gmp_str->unique_neighborList);
    free(gmp_str->X);
    free(gmp_str->dX_dX);
    free(gmp_str->dX_dY);
    free(gmp_str->dX_dZ);
}

void build_gmpObj(GMPObj *gmp_str, NeighList *nlist, FeatureScaler *ftr_scale, int nmcsh, double *atompos, int **params_i, double **params_d, double** atom_gaussian, int* ngaussians, int* element_index_to_order, int* atom_type_to_indices, int* atom_indices){
    
    int cal_num = gmp_str->cal_num, train = 1;
    int i, j, k, N_r;
    int *cal_atoms = gmp_str->cal_atoms, *nneigh = gmp_str->Nneighbors;
    double xi, yi, zi, xj, yj, zj;
    int *ntemp, *imgx_temp, *imgy_temp, *imgz_temp, *elemtyp_temp;
    int idx1, idx2, idx3, idx_neigh, elem_typ, n, l, m;
    double L1, L2, L3;
    double x0,y0,z0,r0_sqr,dtheta,dphi;


    L1 = gmp_str->cell[0];
	L2 = gmp_str->cell[1];
	L3 = gmp_str->cell[2];


    for (int ii=0; ii < cal_num; ++ii) {
        
        i=cal_atoms[ii];       

		xi = atompos[3*i];
		yi = atompos[3*i+1];
		zi = atompos[3*i+2];

		ntemp = (nlist->neighborList +i)->array;
		imgx_temp=(nlist->neighborList_imgX + i)->array;
		imgy_temp=(nlist->neighborList_imgY + i)->array;
		imgz_temp=(nlist->neighborList_imgZ + i)->array;
		elemtyp_temp=(nlist->neighborAtmTyp +i)->array;
	 

        for (int m = 0; m < nmcsh; ++m) {
            int mcsh_order = params_i[m][0], square = params_i[m][1], solid = params_i[m][2];
            int num_groups = get_num_groups(mcsh_order);
            double A = params_d[m][2], alpha = params_d[m][3], inv_rs = params_d[m][5];
            
            double weight = 1.0;
            double sum_square = 0.0;
            
            for (int group_index = 1; group_index < (num_groups+1); ++group_index){
                SolidGMPFunction mcsh_solid_function = get_solid_mcsh_function(mcsh_order, group_index);
                GMPFunction mcsh_function = get_mcsh_function(mcsh_order, group_index);
                double group_coefficient = get_group_coefficients(mcsh_order, group_index);
                int mcsh_type = get_mcsh_type(mcsh_order, group_index);
                
                if (mcsh_type == 1){
                    double sum_desc = 0.0;
					double *sum_dmiu_dxj = (double *) malloc(sizeof(double)*(nneigh[i]+1));
					double *sum_dmiu_dyj = (double *) malloc(sizeof(double)*(nneigh[i]+1));
					double *sum_dmiu_dzj = (double *) malloc(sizeof(double)*(nneigh[i]+1));

					for (int j=0; j<(nneigh[i]+1); j++) {
                        sum_dmiu_dxj[j] = 0.0;
                        sum_dmiu_dyj[j] = 0.0;
                        sum_dmiu_dzj[j] = 0.0;
                    }

                    double m_desc[1], deriv[3];



                    for (int j = -1; j < nneigh[i]; ++j) {

                        int neigh_atom_element_order;
                        if (j < 0) {
                            xj = xi;
                            yj = yi;
                            zj = zi;
                            int neigh_atom_element_index = atom_indices[i];
                            neigh_atom_element_order = element_index_to_order[neigh_atom_element_index];
                        }
                        else{
                            idx_neigh = ntemp[j];
                            elem_typ = elemtyp_temp[j];
                            xj = atompos[3*idx_neigh] + L1 * imgx_temp[j];
                            yj = atompos[3*idx_neigh+1] + L2 * imgy_temp[j];
                            zj = atompos[3*idx_neigh+2] + L3 * imgz_temp[j];
                            int neigh_atom_element_index = atom_type_to_indices[elem_typ];
                            neigh_atom_element_order = element_index_to_order[neigh_atom_element_index];
                        }
						
                        x0 = xj - xi;
                        y0 = yj - yi;
                        z0 = zj - zi;

                        r0_sqr = x0*x0 + y0*y0 + z0*z0;

                        for (int g = 0; g < ngaussians[neigh_atom_element_order]; ++g){
                            double B = atom_gaussian[neigh_atom_element_order][g*2], beta = atom_gaussian[neigh_atom_element_order][g*2+1];
                            if (solid == 1){
                                mcsh_solid_function(x0, y0, z0, r0_sqr, A, B, alpha, beta, m_desc, deriv);
                            } else{
                                mcsh_function(x0, y0, z0, r0_sqr, A, B, alpha, beta, inv_rs, m_desc, deriv);
                            }
                            sum_desc += m_desc[0];
							sum_dmiu_dxj[j+1] += deriv[0];
							sum_dmiu_dyj[j+1] += deriv[1];
							sum_dmiu_dzj[j+1] += deriv[2];
                        }
                    }
                    sum_square += group_coefficient * sum_desc * sum_desc;
					double dmdx, dmdy, dmdz;
                    for (int j = -1; j < nneigh[i]; ++j) {
                        dmdx = (sum_desc * sum_dmiu_dxj[j+1]) * group_coefficient * 2.0;
                        dmdy = (sum_desc * sum_dmiu_dyj[j+1]) * group_coefficient * 2.0;
                        dmdz = (sum_desc * sum_dmiu_dzj[j+1]) * group_coefficient * 2.0;
						if (j < 0) {
							gmp_str->dX_dX[ii][i][m] += dmdx;
							gmp_str->dX_dY[ii][i][m] += dmdy;
							gmp_str->dX_dZ[ii][i][m] += dmdz;
						}
						else {
							gmp_str->dX_dX[ii][(nlist->neighborList[i]).array[j]][m] += dmdx;
							gmp_str->dX_dY[ii][(nlist->neighborList[i]).array[j]][m] += dmdy;
							gmp_str->dX_dZ[ii][(nlist->neighborList[i]).array[j]][m] += dmdz;
						}
                        gmp_str->dX_dX[ii][i][m] -= dmdx;
                        gmp_str->dX_dY[ii][i][m] -= dmdy;
                        gmp_str->dX_dZ[ii][i][m] -= dmdz;						
                    }

                    free(sum_dmiu_dxj);
                    free(sum_dmiu_dyj);
                    free(sum_dmiu_dzj);
                }
                

                if (mcsh_type == 2){
                    double sum_miu1 = 0.0, sum_miu2 = 0.0, sum_miu3 = 0.0;

                    double* sum_dmiu1_dxj = (double *) malloc(sizeof(double)*(nneigh[i]+1));
                    double* sum_dmiu2_dxj = (double *) malloc(sizeof(double)*(nneigh[i]+1));
                    double* sum_dmiu3_dxj = (double *) malloc(sizeof(double)*(nneigh[i]+1));
                    double* sum_dmiu1_dyj = (double *) malloc(sizeof(double)*(nneigh[i]+1));
                    double* sum_dmiu2_dyj = (double *) malloc(sizeof(double)*(nneigh[i]+1));
                    double* sum_dmiu3_dyj = (double *) malloc(sizeof(double)*(nneigh[i]+1));
                    double* sum_dmiu1_dzj = (double *) malloc(sizeof(double)*(nneigh[i]+1));
                    double* sum_dmiu2_dzj = (double *) malloc(sizeof(double)*(nneigh[i]+1));
                    double* sum_dmiu3_dzj = (double *) malloc(sizeof(double)*(nneigh[i]+1));

                    for (int j=0; j<nneigh[i]+1; j++) {
                        sum_dmiu1_dxj[j] = 0.0;
                        sum_dmiu2_dxj[j] = 0.0;
                        sum_dmiu3_dxj[j] = 0.0;
                        sum_dmiu1_dyj[j] = 0.0;
                        sum_dmiu2_dyj[j] = 0.0;
                        sum_dmiu3_dyj[j] = 0.0;
                        sum_dmiu1_dzj[j] = 0.0;
                        sum_dmiu2_dzj[j] = 0.0;
                        sum_dmiu3_dzj[j] = 0.0;
                    }

                    double miu[3], deriv[9];

                    for (int j = -1; j < nneigh[i]; ++j) {
                        int neigh_atom_element_order;
                        if (j < 0) {
                            xj = xi;
                            yj = yi;
                            zj = zi;
                            int neigh_atom_element_index = atom_indices[i];
                            neigh_atom_element_order = element_index_to_order[neigh_atom_element_index];
                        }
                        else{
                            idx_neigh = ntemp[j];
                            elem_typ = elemtyp_temp[j];
                            xj = atompos[3*idx_neigh] + L1 * imgx_temp[j];
                            yj = atompos[3*idx_neigh+1] + L2 * imgy_temp[j];
                            zj = atompos[3*idx_neigh+2] + L3 * imgz_temp[j];
                            int neigh_atom_element_index = atom_type_to_indices[elem_typ];
                            neigh_atom_element_order = element_index_to_order[neigh_atom_element_index];
                        }

                        x0 = xj - xi;
                        y0 = yj - yi;
                        z0 = zj - zi;

                        r0_sqr = x0*x0 + y0*y0 + z0*z0;

                        for (int g = 0; g < ngaussians[neigh_atom_element_order]; ++g){
                            double B = atom_gaussian[neigh_atom_element_order][g*2], beta = atom_gaussian[neigh_atom_element_order][g*2+1];
                            if (solid == 1){
                                mcsh_solid_function(x0, y0, z0, r0_sqr, A, B, alpha, beta, miu, deriv);
                            } else{
                                mcsh_function(x0, y0, z0, r0_sqr, A, B, alpha, beta, inv_rs, miu, deriv);
                            }
                            sum_miu1 += miu[0];
                            sum_miu2 += miu[1];
                            sum_miu3 += miu[2];
							sum_dmiu1_dxj[j+1] += deriv[0];
							sum_dmiu1_dyj[j+1] += deriv[1];
                            sum_dmiu1_dzj[j+1] += deriv[2];
                            sum_dmiu2_dxj[j+1] += deriv[3];
                            sum_dmiu2_dyj[j+1] += deriv[4];
                            sum_dmiu2_dzj[j+1] += deriv[5];
                            sum_dmiu3_dxj[j+1] += deriv[6];
                            sum_dmiu3_dyj[j+1] += deriv[7];
                            sum_dmiu3_dzj[j+1] += deriv[8];
                        }
					}
                    sum_square += group_coefficient * (sum_miu1*sum_miu1 + sum_miu2*sum_miu2 + sum_miu3*sum_miu3);
					
					double dmdx, dmdy, dmdz;
					for (int j = -1; j < nneigh[i]; ++j) {
                        dmdx = (sum_miu1 * sum_dmiu1_dxj[j+1] + sum_miu2 * sum_dmiu2_dxj[j+1] + sum_miu3 * sum_dmiu3_dxj[j+1]) * group_coefficient * 2.0;
                        dmdy = (sum_miu1 * sum_dmiu1_dyj[j+1] + sum_miu2 * sum_dmiu2_dyj[j+1] + sum_miu3 * sum_dmiu3_dyj[j+1]) * group_coefficient * 2.0;
                        dmdz = (sum_miu1 * sum_dmiu1_dzj[j+1] + sum_miu2 * sum_dmiu2_dzj[j+1] + sum_miu3 * sum_dmiu3_dzj[j+1]) * group_coefficient * 2.0;

                        if (j < 0) {
							gmp_str->dX_dX[ii][i][m] += dmdx;
							gmp_str->dX_dY[ii][i][m] += dmdy;
							gmp_str->dX_dZ[ii][i][m] += dmdz;
						}
						else {
							gmp_str->dX_dX[ii][(nlist->neighborList[i]).array[j]][m] += dmdx;
							gmp_str->dX_dY[ii][(nlist->neighborList[i]).array[j]][m] += dmdy;
							gmp_str->dX_dZ[ii][(nlist->neighborList[i]).array[j]][m] += dmdz;
						}

                        gmp_str->dX_dX[ii][i][m] -= dmdx;
                        gmp_str->dX_dY[ii][i][m] -= dmdy;
                        gmp_str->dX_dZ[ii][i][m] -= dmdz;
                    }

                    free(sum_dmiu1_dxj);
                    free(sum_dmiu1_dyj);
                    free(sum_dmiu1_dzj);
					free(sum_dmiu2_dxj);
                    free(sum_dmiu2_dyj);
                    free(sum_dmiu2_dzj);
					free(sum_dmiu3_dxj);
                    free(sum_dmiu3_dyj);
                    free(sum_dmiu3_dzj);
                }

                if (mcsh_type == 3){
                    double sum_miu1 = 0.0, sum_miu2 = 0.0, sum_miu3 = 0.0, sum_miu4 = 0.0, sum_miu5 = 0.0, sum_miu6 = 0.0;

					double* sum_dmiu1_dxj = (double *) malloc(sizeof(double)*(nneigh[i]+1));
                    double* sum_dmiu2_dxj = (double *) malloc(sizeof(double)*(nneigh[i]+1));
                    double* sum_dmiu3_dxj = (double *) malloc(sizeof(double)*(nneigh[i]+1));
                    double* sum_dmiu4_dxj = (double *) malloc(sizeof(double)*(nneigh[i]+1));
                    double* sum_dmiu5_dxj = (double *) malloc(sizeof(double)*(nneigh[i]+1));
                    double* sum_dmiu6_dxj = (double *) malloc(sizeof(double)*(nneigh[i]+1));
                    double* sum_dmiu1_dyj = (double *) malloc(sizeof(double)*(nneigh[i]+1));
                    double* sum_dmiu2_dyj = (double *) malloc(sizeof(double)*(nneigh[i]+1));
                    double* sum_dmiu3_dyj = (double *) malloc(sizeof(double)*(nneigh[i]+1));
                    double* sum_dmiu4_dyj = (double *) malloc(sizeof(double)*(nneigh[i]+1));
                    double* sum_dmiu5_dyj = (double *) malloc(sizeof(double)*(nneigh[i]+1));
                    double* sum_dmiu6_dyj = (double *) malloc(sizeof(double)*(nneigh[i]+1));
                    double* sum_dmiu1_dzj = (double *) malloc(sizeof(double)*(nneigh[i]+1));
                    double* sum_dmiu2_dzj = (double *) malloc(sizeof(double)*(nneigh[i]+1));
                    double* sum_dmiu3_dzj = (double *) malloc(sizeof(double)*(nneigh[i]+1));
                    double* sum_dmiu4_dzj = (double *) malloc(sizeof(double)*(nneigh[i]+1));
                    double* sum_dmiu5_dzj = (double *) malloc(sizeof(double)*(nneigh[i]+1));
                    double* sum_dmiu6_dzj = (double *) malloc(sizeof(double)*(nneigh[i]+1));

					for (int j=0; j<nneigh[i]+1; j++) {
                        sum_dmiu1_dxj[j] = 0.0;
                        sum_dmiu2_dxj[j] = 0.0;
                        sum_dmiu3_dxj[j] = 0.0;
                        sum_dmiu4_dxj[j] = 0.0;
                        sum_dmiu5_dxj[j] = 0.0;
                        sum_dmiu6_dxj[j] = 0.0;
                        sum_dmiu1_dyj[j] = 0.0;
                        sum_dmiu2_dyj[j] = 0.0;
                        sum_dmiu3_dyj[j] = 0.0;
                        sum_dmiu4_dyj[j] = 0.0;
                        sum_dmiu5_dyj[j] = 0.0;
                        sum_dmiu6_dyj[j] = 0.0;
                        sum_dmiu1_dzj[j] = 0.0;
                        sum_dmiu2_dzj[j] = 0.0;
                        sum_dmiu3_dzj[j] = 0.0;
                        sum_dmiu4_dzj[j] = 0.0;
                        sum_dmiu5_dzj[j] = 0.0;
                        sum_dmiu6_dzj[j] = 0.0;
					}
					
                    double miu[6], deriv[18];
                    int neigh_atom_element_order;
                    for (int j = -1; j < nneigh[i]; ++j) {
                        if (j < 0) {
                            xj = xi;
                            yj = yi;
                            zj = zi;
                            int neigh_atom_element_index = atom_indices[i];
                            neigh_atom_element_order = element_index_to_order[neigh_atom_element_index];
                        }
                        else{
                            idx_neigh = ntemp[j];
                            elem_typ = elemtyp_temp[j];
                            xj = atompos[3*idx_neigh] + L1 * imgx_temp[j];
                            yj = atompos[3*idx_neigh+1] + L2 * imgy_temp[j];
                            zj = atompos[3*idx_neigh+2] + L3 * imgz_temp[j];
                            int neigh_atom_element_index = atom_type_to_indices[elem_typ];
                            neigh_atom_element_order = element_index_to_order[neigh_atom_element_index];
                        }

                        x0 = xj - xi;
                        y0 = yj - yi;
                        z0 = zj - zi;

                        r0_sqr = x0*x0 + y0*y0 + z0*z0;

                        for (int g = 0; g < ngaussians[neigh_atom_element_order]; ++g){
                            double B = atom_gaussian[neigh_atom_element_order][g*2], beta = atom_gaussian[neigh_atom_element_order][g*2+1];
                            if (solid == 1){
                                mcsh_solid_function(x0, y0, z0, r0_sqr, A, B, alpha, beta, miu, deriv);
                            } else{
                                mcsh_function(x0, y0, z0, r0_sqr, A, B, alpha, beta, inv_rs, miu, deriv);
                            }
                            sum_miu1 += miu[0];
                            sum_miu2 += miu[1];
                            sum_miu3 += miu[2];
                            sum_miu4 += miu[3];
                            sum_miu5 += miu[4];
                            sum_miu6 += miu[5];
							sum_dmiu1_dxj[j+1] += deriv[0];
                            sum_dmiu1_dyj[j+1] += deriv[1];
                            sum_dmiu1_dzj[j+1] += deriv[2];
                            sum_dmiu2_dxj[j+1] += deriv[3];
                            sum_dmiu2_dyj[j+1] += deriv[4];
                            sum_dmiu2_dzj[j+1] += deriv[5];
                            sum_dmiu3_dxj[j+1] += deriv[6];
                            sum_dmiu3_dyj[j+1] += deriv[7];
                            sum_dmiu3_dzj[j+1] += deriv[8];
                            sum_dmiu4_dxj[j+1] += deriv[9];
                            sum_dmiu4_dyj[j+1] += deriv[10];
                            sum_dmiu4_dzj[j+1] += deriv[11];
                            sum_dmiu5_dxj[j+1] += deriv[12];
                            sum_dmiu5_dyj[j+1] += deriv[13];
                            sum_dmiu5_dzj[j+1] += deriv[14];
                            sum_dmiu6_dxj[j+1] += deriv[15];
                            sum_dmiu6_dyj[j+1] += deriv[16];
                            sum_dmiu6_dzj[j+1] += deriv[17];
                        }
                    }
                    sum_square += group_coefficient * (sum_miu1*sum_miu1 + sum_miu2*sum_miu2 + sum_miu3*sum_miu3 +
                                                    sum_miu4*sum_miu4 + sum_miu5*sum_miu5 + sum_miu6*sum_miu6);
					
					double dmdx, dmdy, dmdz;

                    for (int j = -1; j < nneigh[i]; ++j) {
                        dmdx = (sum_miu1 * sum_dmiu1_dxj[j+1] + sum_miu2 * sum_dmiu2_dxj[j+1] +
                                sum_miu3 * sum_dmiu3_dxj[j+1] + sum_miu4 * sum_dmiu4_dxj[j+1] +
                                sum_miu5 * sum_dmiu5_dxj[j+1] + sum_miu6 * sum_dmiu6_dxj[j+1]) * group_coefficient * 2.0;

                        dmdy = (sum_miu1 * sum_dmiu1_dyj[j+1] + sum_miu2 * sum_dmiu2_dyj[j+1] +
                                sum_miu3 * sum_dmiu3_dyj[j+1] + sum_miu4 * sum_dmiu4_dyj[j+1] +
                                sum_miu5 * sum_dmiu5_dyj[j+1] + sum_miu6 * sum_dmiu6_dyj[j+1]) * group_coefficient * 2.0;

                        dmdz = (sum_miu1 * sum_dmiu1_dzj[j+1] + sum_miu2 * sum_dmiu2_dzj[j+1] +
                                sum_miu3 * sum_dmiu3_dzj[j+1] + sum_miu4 * sum_dmiu4_dzj[j+1] +
                                sum_miu5 * sum_dmiu5_dzj[j+1] + sum_miu6 * sum_dmiu6_dzj[j+1]) * group_coefficient * 2.0;

						if (j < 0) {
							gmp_str->dX_dX[i][i][m] += dmdx;
							gmp_str->dX_dY[i][i][m] += dmdy;
							gmp_str->dX_dZ[i][i][m] += dmdz;
						}
						else {
							gmp_str->dX_dX[i][(nlist->neighborList[i]).array[j]][m] += dmdx;
							gmp_str->dX_dY[i][(nlist->neighborList[i]).array[j]][m] += dmdy;
							gmp_str->dX_dZ[i][(nlist->neighborList[i]).array[j]][m] += dmdz;
						}

                        gmp_str->dX_dX[i][i][m] -= dmdx;
                        gmp_str->dX_dY[i][i][m] -= dmdy;
                        gmp_str->dX_dZ[i][i][m] -= dmdz;
					}
                    free(sum_dmiu1_dxj);
                    free(sum_dmiu2_dxj);
                    free(sum_dmiu3_dxj);
                    free(sum_dmiu4_dxj);
                    free(sum_dmiu5_dxj);
                    free(sum_dmiu6_dxj);
                    free(sum_dmiu1_dyj);
                    free(sum_dmiu2_dyj);
                    free(sum_dmiu3_dyj);
                    free(sum_dmiu4_dyj);
                    free(sum_dmiu5_dyj);
                    free(sum_dmiu6_dyj);
                    free(sum_dmiu1_dzj);
                    free(sum_dmiu2_dzj);
                    free(sum_dmiu3_dzj);
                    free(sum_dmiu4_dzj);
                    free(sum_dmiu5_dzj);
                    free(sum_dmiu6_dzj);

                }
            }

            if (square != 0){
                gmp_str->X[ii][m] = sum_square;
            }
            else {
                gmp_str->X[ii][m] = sqrt(sum_square);
            }

        }
    }

	// if (train == 0){
	// 	scale_features(gmp_str, ftr_scale, nmcsh);
	// }
	// else {
	// 	ftr_scale->mean_colWise = (double *) malloc(sizeof(double)*nmcsh);
	// 	ftr_scale->stdev_colWise = (double *) malloc(sizeof(double)*nmcsh);
	// 	double means[nmcsh], sums_sq[nmcsh];
	// 	for (int i = 0; i < nmcsh; i++){
	// 		means[i] = 0.;
	// 		sums_sq[i] = 0.;
	// 	}
	// 	for (int i = 0; i < cal_num; i++){
	// 		for (int j = 0; j < nmcsh; j++){
	// 			means[j] += (gmp_str->X[i][j])/cal_num;
	// 		}
	// 	}
	// 	for (int i = 0; i < cal_num; i++){
	// 		for (int j = 0; j < nmcsh; j++){
	// 			sums_sq[j] += pow((gmp_str->X[i][j])-means[j],2);
	// 		}
	// 	}
	// 	for (int i = 0; i < nmcsh; i++){
	// 		ftr_scale->mean_colWise[i] = means[i];
	// 		double tmp_stdev = sqrt(sums_sq[i]/cal_num);
	// 		if (tmp_stdev < .005) {
	// 			ftr_scale->stdev_colWise[i] = 1.;
	// 		} else {
	// 			ftr_scale->stdev_colWise[i] = tmp_stdev;
	// 		}
	// 	}
	// 	scale_features(gmp_str, ftr_scale, nmcsh);
	// }

}

void scale_features(GMPObj *gmp_str, FeatureScaler *ftr_scale, int nmcsh){

	int cal_num = gmp_str->cal_num;
	int atom_num = gmp_str->natom;

	for (int i = 0; i < cal_num; i++){
		for (int j = 0; j < nmcsh; j++){
			double desc_entry = gmp_str->X[i][j], mean = ftr_scale->mean_colWise[j], stdev = ftr_scale->stdev_colWise[j];
			gmp_str->X[i][j] = (desc_entry-mean)/stdev;
			for (int k = 0; k < atom_num; k++){
				double dmdx = gmp_str->dX_dX[i][k][j], dmdy = gmp_str->dX_dY[i][k][j], dmdz = gmp_str->dX_dZ[i][k][j];
				gmp_str->dX_dX[i][k][j] = dmdx/stdev, gmp_str->dX_dY[i][k][j] = dmdy/stdev, gmp_str->dX_dZ[i][k][j] = dmdz/stdev;
			}
		}
	}
}