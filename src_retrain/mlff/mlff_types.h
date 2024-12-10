#ifndef MLFF_TYPES_H
#define MLFF_TYPES_H

#include "ddbp_types.h"


#define L_STRING 512
// typedef int value_type;
// /*
//  * @brief  Data type for dynamic array.
//  */
// typedef struct {
//     value_type *array;
//     size_t len;  // length of the array (used)
//     size_t capacity; // total capacity of memory available
// } dyArray;


typedef struct NeighList NeighList;
typedef struct SoapObj SoapObj;
typedef struct MLFF_Obj MLFF_Obj;

typedef struct NeighList
{
  // # of atoms 
  int natom;
  // # of element species
  int nelem;
  // cutoff distance (Bohr)
  double rcut;
  // length of fundamental unit cell
  // Cell description
  int cell_typ;
  int BC[3];
  double cell_len[3];
  double LatUVec[9];
  double twist;
  // # of atom for each eleemnt species

  int natom_domain;
  int *atom_idx_domain;
  int *el_idx_domain;


  int* natom_elem;
  // # of neighbours for each atoms
  int* Nneighbors;
  // # of neighbour in the fundamental cell for each atom (excludes neighbours in periodic images)
  int* unique_Nneighbors;
  // # of neighbours of a given element speciy for each atoms
  int** Nneighbors_elemWise;
  // # of neigbours of a given element speciy in the fundamental cell for each atom (excludes neighbours in periodic images) 
  int** unique_Nneighbors_elemWise;
  // list of neighbours of all atoms
  int *if_self_image;
  // if the neighbor list of an atom contains it's self image
  dyArray* neighborList;
  // list of neighbours of a given element speciy
  dyArray** neighborList_elemWise;
  // list of unique neighbours of all atoms
  dyArray* unique_neighborList;
  // list of unique neighbous of a given speciy
  dyArray** unique_neighborList_elemWise;
  // list of the image ID in x-direction of neighbours for all atoms
  dyArray* neighborList_imgX;
  // list of the image ID in y-direction of neighbours for all atoms
  dyArray* neighborList_imgY;
  // list of the image ID in z-direction of neighbours for all atoms
  dyArray* neighborList_imgZ;
  // list of the image ID in x-direction of neighbours of a given element speciy for all atoms
  dyArray** neighborList_elemWise_imgX;
  // list of the image ID in y-direction of neighbours of a given element speciy for all atoms
  dyArray** neighborList_elemWise_imgY;
  // list of the image ID in z-direction of neighbours of a given element speciy for all atoms
  dyArray** neighborList_elemWise_imgZ;
  // stores an ID of all the neighbours of the atoms. [IDs are integers ranging from 0 to N-1 whre N is number of unique neighbours of the concerning atom]
  dyArray* localID_neighbours;
  // stores an ID of all the unique neighbours of the atoms. [IDs are integers ranging from 0 to N-1 whre N is number of unique neighbours of the concerning atom]
  dyArray** localID_neighbours_elem;
  // list atom type of neighbours of a given atom
  dyArray* neighborAtmTyp;
} NeighList;


typedef struct SoapObj
{
  // # of atoms 
  int natom;
  // cutoff distance (Bohr)
  double rcut;
  // dimensions of the fundamental unit cell (Bohr)
  //double cell_len[3];
  double cell_measure;

  int natom_domain;
  int *atom_idx_domain;
  int *el_idx_domain;
  int *if_self_image;
  // if the neighbor list of an atom contains it's self image

  // Max angular momentum of Spherical harmonics in SOAP
  int Lmax;
  // Number of radial basis functions (spherical Bessel)
  int Nmax;
  // size of the 2-body descriptor
  int size_X2;
  // size of the 3-body angular descriptor
  int size_X3;
  // weight of the two body descriptor in SOAP kernel (1-beta_3)
  double beta_2;
  // weight of the three body descriptor in SOAP kernel
  double beta_3;
  // exponent in the dot product kernel of SOAP
  double xi_3;
  // # of elements
  int nelem;
  // # number of grids points used in the spline interpolation of h_nl
  int N_rgrid;
  // array of number of neighbours for all the atoms
  int* Nneighbors;

  // array of number of unqiue neighbours for all the atoms
  int* unique_Nneighbors;
  // array of number of atoms of different element species 
  int* natom_elem;
  // array of number of neighbours of different species for all the atoms
  int** unique_Nneighbors_elemWise;
  // list of neighbours of different species in the fundamental cell
  dyArray** unique_neighborList_elemWise;
  // list of neighbours of all the atoms
  dyArray* neighborList;
  // list of unique neighbours of all the atoms
  dyArray* unique_neighborList;
  // array of 2-body descriptor, X2[natom][size_X2]
  double **X2;
  // array of 3-body descriptor SOAP, X3[natom][size_X3]
  double **X3;
  // array of 2-body descriptor derivatives for stress, dX2_dF[natom][6][size_X2]
  double ***dX2_dF;
  // array of 3-body descriptor derivatives for stress, dX2_dF[natom][6][size_X3]
  double ***dX3_dF;
  // array of 2-body descriptor derivatives for x-dir force, dX2_dX[natom][1+neigh][size_X2]
  double ***dX2_dX;
  // array of 2-body descriptor derivatives for y-dir force, dX2_dY[natom][1+neigh][size_X2]
  double ***dX2_dY;
  // array of 2-body descriptor derivatives for z-dir force, dX2_dZ[natom][1+neigh][size_X2]
  double ***dX2_dZ;
  // array of 3-body descriptor derivatives for x-dir force, dX3_dX[natom][1+neigh][size_X3]
  double ***dX3_dX;
  // array of 3-body descriptor derivatives for y-dir force, dX3_dY[natom][1+neigh][size_X3]
  double ***dX3_dY;
  // array of 3-body descriptor derivatives for z-dir force, dX3_dZ[natom][1+neigh][size_X3]
  double ***dX3_dZ;
} SoapObj;




typedef struct MLFF_Obj
{

  int natom_domain;
  int *atom_idx_domain;
  int *el_idx_domain;
  int *Znucl;
  double condK_min;
  int if_sparsify_before_train;
  // Design matrix built during the training [rows: (N_str*(3*natom+6+1)), cols: N_train] (updated after every training step)
  // N_str is the number of reference structures in the training dataset, N_train is the number of local confiugrations in the training dataset
  double **K_train;
  // vector storing the energy, forces and stress of all the reference structure in the training dataset [size: N_str*(3*natom+6+1))]
  double *b_no_norm;
  // weights in the linear model [size: N_train]
  double *AtA_SVD_U;
  double *AtA_SVD_Vt;
  double *AtA_SingVal;
  double *AtA;

  int print_mlff_flag;
  
  double *weights;
  // Covariance matrix learned from the training data in ColMajor [row: N_train, cols: N_train]
  double *cov_train;
  // number of different speciy of elements in the system
  // int MD_iter;
  dyArray atom_idx_addtrain;
  int nelem;
  // int if_only_mlff;
  // number of atoms in the system
   int natom;
  // number of local descriptors of various elements in the training dataset, [size: nelem]  
  int *natm_train_elemwise;   
  int descriptor_typ;
  // total number of local descriptors in the training dataset
  int natm_train_total;
  // Array containing the element type of local descriptors in the training dataset
  int *natm_typ_train;
  // Number of reference structure in the training dataset
  int n_str;
  // maximum number of reference structure allowed in the training dataset
  int n_str_max;
  // maximum number of local descriptors allowed in the training dataset (per element type)
  int n_train_max;
  // basis set size for the SOAP
  int Nmax;
  int Lmax;
  int kernel_typ;
  // tolerance of force error to decide to skip DFT
  double F_tol;
  // length of 2-body descriptor in SOAP
  int size_X2;
  // length of 3-body descriptor in SOAP
  int size_X3;
  // number of rows in the design matrix
  int n_rows;
  // number of columns in the design matrix
  int n_cols;
  // number of radial points
  int N_rgrid;
  // weights for 2-body terms in the SOAP kernel
  double error_train_E;
  double error_train_F;
  double *error_train_stress;
  double beta_2;
  // weights for 3-body terms in the SOAP kernel
  double beta_3;
  // exponent of dot product in the SOAP kernel
  double sigma_atom;

  int *E_row_idx;
  
  double xi_3;
  // cutoff radius for SOAP
  double rcut;
  // mean of atomic energies of all reference structures
  double mu_E;
  // std deviation of atomic energies of all reference structures
  double std_E;
  // std deviation of atomic energies of all reference structures
  double std_F;
  // hyperparameters in the regression
  double sigma_w;
  double sigma_v;
  // scaling to be applied on the design and prediction matrices
  double E_scale;
  double F_scale;
  double *stress_scale;
  double relative_scale_F;
  double *relative_scale_stress;
  
  double *F_store;
  double *E_store;
  double **stress_store;
  // double *stress1_store;
  // double *stress2_store;
  // double *stress3_store;
  // double *stress4_store;
  // double *stress5_store;


  int E_store_counter;
  int F_store_counter;
  int *variable_type_identifier;

  // mean of stress tensor of all reference structures
  //double *mu_stress;
  // std deviation of stress tensor of all reference structures
  double *std_stress;
  // stores 2-body SOAP descriptor for all local descriptors in the training dataset
  double **X2_traindataset;
  // stores 3-body SOAP descriptor for all local descriptors in the training dataset
  double **X3_traindataset;
  // stores SOAP descriptors and other related variables for all reference structures in the training dataset
  // ********** Memory Bottleneck *****************
  SoapObj *soap_descriptor_strdataset;
  // precalculated values of h_nl and its derivative on a sparse grid. Later used in spline interpolation.
  double* rgrid;
  double* h_nl;
  double* dh_nl;
  char ref_atom_name[L_STRING];
  char ref_str_name[L_STRING];
  char restart_name[L_STRING];
  int stress_len;
  int mlff_flag;
  FILE *fp_mlff;
} MLFF_Obj;

#endif 
