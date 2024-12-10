#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#include "md.h"
#include "isddft.h"
#include "orbitalElecDensInit.h"
#include "initialization.h"
#include "electronicGroundState.h"
#include "stress.h"
#include "tools.h"
#include "pressure.h"
#include "relax.h"
#include "electrostatics.h"
#include "eigenSolver.h" // Mesh2ChebDegree
#include "readfiles.h"
#include "ofdft.h"
#include "parallelization.h"
#include "krylovschur.h"
#include "inverse.h"
#include "electrostatics.h"
#include "exchangeCorrelation.h"
#include "relax.h"
#include "lapVecRoutines.h"
#include "cyclix_tools.h"
#include "gradVecRoutines.h"

#define SIGN(a,b) ((b)>=(0)?fabs(a):-fabs(a))
#define max(a,b) ((a)>(b)?(a):(b))

void main_INVERSE(SPARC_OBJ *pSPARC)
{
	int rank;
	int *DMVertices;
	int DMnd;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm comm;
    comm = pSPARC->dmcomm_phi;
    pSPARC->OFDFT_Cf = 0.3*pow(3*M_PI*M_PI,2.0/3);
    pSPARC->OFDFT_lambda = 1.0;
    pSPARC->count_invert_derivative = 0;

    DMnd = pSPARC->Nd_d;
    DMVertices = pSPARC->DMVertices;

    int gridsizes[3];
    gridsizes[0] = pSPARC->Nx;
    gridsizes[1] = pSPARC->Ny;
    gridsizes[2] = pSPARC->Nz;




    pSPARC->rho_data_inverse = (double *) malloc(sizeof(double)* DMnd);
    pSPARC->dL_dVp = (double *) malloc(sizeof(double)* DMnd);


	read_vec(pSPARC->rho_data_inverse, gridsizes, DMVertices, 1, pSPARC->fname_rho_data, comm);

    GetInfluencingAtoms(pSPARC);
    Generate_PseudoChargeDensity(pSPARC);


	pSPARC->Vxc_data_inverse = (double *) malloc(sizeof(double)* DMnd);
	pSPARC->phi_data_inverse = (double *) malloc(sizeof(double)* DMnd);
	pSPARC->VTF_data_inverse = (double *) malloc(sizeof(double)* DMnd);

	Calculate_Vxc(pSPARC);
	Calculate_elecstPotential(pSPARC);

	double factor_Vxc = 0.0, factor_VTF = 0.0;

	if (pSPARC->Vxc_flag_inverse) {
		factor_Vxc = 1.0;
	}
	if (pSPARC->VTF_flag_inverse) {
		factor_VTF = 1.0;
	}

	for (int i = 0; i < DMnd; i++){
		pSPARC->Vxc_data_inverse[i] = factor_Vxc * pSPARC->XCPotential[i];
		pSPARC->phi_data_inverse[i] = pSPARC->elecstPotential[i];
		pSPARC->VTF_data_inverse[i] = factor_VTF * (5.0 / 3.0) * pSPARC->OFDFT_Cf * pow(pSPARC->rho_data_inverse[i], (2.0/3));
	}


    

	pSPARC->Vp_inverse = (double *) malloc(sizeof(double)* DMnd);

	if (pSPARC->read_Vp_guess_flag_inverse == 0){
		for (int i = 0; i < DMnd; i++){
			pSPARC->Vp_inverse[i] = 0.0;
		}
	} else if (pSPARC->read_Vp_guess_flag_inverse == 1){
		SeededRandVec (pSPARC->Vp_inverse, DMVertices, gridsizes, 0.0, 1.0, 0);
	} else if (pSPARC->read_Vp_guess_flag_inverse == 2){
		read_vec(pSPARC->Vp_inverse, gridsizes, DMVertices, 0, pSPARC->fname_Vp_guess, comm);
	}

    double *Drho_x = (double *) malloc(DMnd * sizeof(double));
    double *Drho_y = (double *) malloc(DMnd * sizeof(double));
    double *Drho_z = (double *) malloc(DMnd * sizeof(double));

    double *DDrho_x = (double *) malloc(DMnd * sizeof(double));
    double *DDrho_y = (double *) malloc(DMnd * sizeof(double));
    double *DDrho_z = (double *) malloc(DMnd * sizeof(double));



    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, pSPARC->rho_data_inverse, Drho_x, 0, pSPARC->dmcomm_phi);
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, pSPARC->rho_data_inverse, Drho_y, 1, pSPARC->dmcomm_phi);
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, pSPARC->rho_data_inverse, Drho_z, 2, pSPARC->dmcomm_phi);

    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, Drho_x, DDrho_x, 0, pSPARC->dmcomm_phi);
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, Drho_y, DDrho_y, 1, pSPARC->dmcomm_phi);
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, Drho_z, DDrho_z, 2, pSPARC->dmcomm_phi);

    char *fname = malloc(5000);
    strcpy(fname, "grad_rho_data_x.txt");
    print_vec(Drho_x, gridsizes, DMVertices, fname, comm);
    strcpy(fname, "grad_rho_data_y.txt");
    print_vec(Drho_y, gridsizes, DMVertices, fname, comm);
    strcpy(fname, "grad_rho_data_z.txt");
    print_vec(Drho_z, gridsizes, DMVertices, fname, comm);

    strcpy(fname, "lap_rho_data_x.txt");
    print_vec(DDrho_x, gridsizes, DMVertices, fname, comm);
    strcpy(fname, "lap_rho_data_y.txt");
    print_vec(DDrho_y, gridsizes, DMVertices, fname, comm);
    strcpy(fname, "lap_rho_data_z.txt");
    print_vec(DDrho_z, gridsizes, DMVertices, fname, comm);

    strcpy(fname, "V_xc_data.txt");
    print_vec(pSPARC->Vxc_data_inverse, gridsizes, DMVertices, fname, comm);

    strcpy(fname, "V_TF_data.txt");
    print_vec(pSPARC->VTF_data_inverse, gridsizes, DMVertices, fname, comm);
    free(fname);

    // FILE *fp;
    // if (rank==0){
    //    fp = fopen("grad_rho_data.txt","w");
    //     for (int i = 0; i < DMnd; i++){
    //         fprintf(fp,"%.14f %.14f %.14f\n",Drho_x[i], Drho_y[i], Drho_z[i]);
    //     }
    //     fclose(fp); 

    //     fp = fopen("Lap_rho_data.txt","w");
    //     for (int i = 0; i < DMnd; i++){
    //         fprintf(fp,"%.14f %.14f %.14f\n",DDrho_x[i], DDrho_y[i], DDrho_z[i]);
    //     }
    //     fclose(fp); 

    //     fp = fopen("V_xc_data.txt","w");
    //     for (int i = 0; i < DMnd; i++){
    //         fprintf(fp,"%.14f\n",pSPARC->Vxc_data_inverse[i]);
    //     }
    //     fclose(fp); 

    //     fp = fopen("V_TF_data.txt","w");
    //     for (int i = 0; i < DMnd; i++){
    //         fprintf(fp,"%.14f\n",pSPARC->VTF_data_inverse[i]);
    //     }
    //     fclose(fp); 

    // }


    free(Drho_x);
    free(Drho_y);
    free(Drho_z);

    free(DDrho_x);
    free(DDrho_y);
    free(DDrho_z);
 



    double Vp_shift = 0.0;
    VectorSum(pSPARC->Vp_inverse, DMnd, &Vp_shift, pSPARC->dmcomm_phi);
    Vp_shift /= (double)pSPARC->Nd;
    VectorShift(pSPARC->Vp_inverse, DMnd, -Vp_shift, pSPARC->dmcomm_phi);


    LBFGS_inverse(pSPARC);

}



/**
 * @brief   Calculate Hamiltonian times a vector in a matrix-free way.
 *          
 *          The Hamiltonian includes the TFW kinetic functional. 
 *          TODO: add WGC kinetic functional. 
 */
void HamiltonianVecRoutines_OFDFT_inverse(
        SPARC_OBJ *pSPARC, int DMnd, int *DMVertices,
        double *u, double *Hu, MPI_Comm comm) {
    
    int i, nproc;
    double cst;
    MPI_Comm_size(comm, &nproc);

    // for (i = 0; i < DMnd; i ++)
    //     pSPARC->electronDens[i] = u[i] * u[i];
    
    // solve the poisson equation for electrostatic potential, "phi"
    // Calculate_elecstPotential(pSPARC);
    
    // calculate xc potential (LDA, PW92), "Vxc"
    // Calculate_Vxc(pSPARC);
    
    // calculate Veff_loc_dmcomm_phi = phi + Vxc in "phi-domain"
    // Calculate_Veff_loc_dmcomm_phi(pSPARC);

    // calculate Vk TFW kinetic functional
    // Vk = (5/3)*Cf*(rho.^(2/3))
    // cst = (5.0 / 3.0) * pSPARC->OFDFT_Cf;
    // for (i = 0; i < DMnd; i++) 
    //     pSPARC->Veff_loc_dmcomm_phi[i] +=  cst * pow(pSPARC->electronDens[i], (2.0/3));

    for (i = 0; i < DMnd; i++) {
        pSPARC->Veff_loc_dmcomm_phi[i] =  pSPARC->Vxc_data_inverse[i] + pSPARC->phi_data_inverse[i] + pSPARC->VTF_data_inverse[i] + pSPARC->Vp_inverse[i];
    }



    int dims[3], periods[3], my_coords[3];
    if (nproc > 1)
        MPI_Cart_get(comm, 3, dims, periods, my_coords);
    else 
        dims[0] = dims[1] = dims[2] = 1;
    
    cst = -0.5 * pSPARC->OFDFT_lambda;

    if (pSPARC->cell_typ == 0)
        Lap_plus_diag_vec_mult_orth(
            pSPARC, DMnd, DMVertices, 1, cst, 1.0, 0.0, 
            pSPARC->Veff_loc_dmcomm_phi, u, Hu, comm, dims);
    else
        Lap_plus_diag_vec_mult_nonorth(
            pSPARC, DMnd, DMVertices, 1, cst, 1.0, 0.0,
            pSPARC->Veff_loc_dmcomm_phi, u, Hu, comm, pSPARC->comm_dist_graph_phi, dims);
    return;
}


void linear_system_Ax_inverse(SPARC_OBJ *pSPARC, int DMnd, int *DMVertices, double epsilon, double *u,
        double *x, double *Ax, MPI_Comm comm)
{   
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    for (int i = 0; i < DMnd; i++) {
        Ax[i] =  0.0;
    }

    if (rank == pSPARC->rank_last_proc_inverse){
        Ax[DMnd] = 0.0;
    }

	HamiltonianVecRoutines_OFDFT_inverse(
       pSPARC, DMnd, DMVertices,
        x, Ax, comm);



    
    double x_end_term_global;
    if (rank==pSPARC->rank_last_proc_inverse){
        x_end_term_global = x[DMnd];
    }
    MPI_Bcast( &x_end_term_global, 1, MPI_DOUBLE, pSPARC->rank_last_proc_inverse, 
               MPI_COMM_WORLD );
    

	for (int i = 0; i < DMnd; i++) {
		Ax[i] = (Ax[i] - epsilon * x[i] + x_end_term_global*u[i]);
	}

    double local_xu = 0.0;
    for (int i = 0; i < DMnd; i++) {
        local_xu =  local_xu + u[i]*x[i];
    }
    MPI_Reduce(&local_xu, &Ax[DMnd], 1, MPI_DOUBLE, MPI_SUM, pSPARC->rank_last_proc_inverse, MPI_COMM_WORLD);




    // if (rank == pSPARC->rank_last_proc_inverse){
    //     for (int i = 0; i < DMnd; i++) {
    //         Ax[DMnd] =  Ax[DMnd] + u[i]*x[i];
    //     }
    // }
	
	
}





void Calculate_Inversion_derivative(SPARC_OBJ *pSPARC)
{	
    int  rank, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);


	int *DMVertices;
	DMVertices = pSPARC->DMVertices;
	double eigmin;
	double *eigvec_min;
	int DMnd;
	double *b, *x0, *x, *x0_linearsolve;

	MPI_Comm comm;
    comm = pSPARC->dmcomm_phi;


	DMnd = pSPARC->Nd_d;
	eigvec_min = (double *) malloc(sizeof(double)*DMnd);
	x0 = (double *) malloc(sizeof(double)*DMnd);



	if (pSPARC->count_invert_derivative ==0) {
		for (int i = 0; i < DMnd; i++){
			x0[i] = sqrt(pSPARC->rho_data_inverse[i]);
		}
	} else {
		for (int i = 0; i < DMnd; i++){
			x0[i] = sqrt(pSPARC->electronDens[i]);
		}
	}

    int nmax = 20;
    int nmin = 10;

	krylovschur_min(pSPARC, DMVertices,
             &eigmin, eigvec_min, x0, pSPARC->KRYLOV_SCHUR_TOL_inverse,
             pSPARC->MAXITER_KRYLOV_SCHUR_inverse, nmax, nmin, 1);

    



	for (int i = 0; i < DMnd; i++){
		eigvec_min[i] =  eigvec_min[i] * sqrt((pSPARC->Nelectron / pSPARC->dV));
		pSPARC->electronDens[i] = eigvec_min[i] * eigvec_min[i];
	}

    double *rho_error;
    double norm_error, norm_rho_data, relative_norm_error;
    rho_error = (double *)malloc(sizeof(double)*DMnd);

    for (int i = 0; i < DMnd; i++){
        rho_error[i] = pSPARC->rho_data_inverse[i] -  pSPARC->electronDens[i];
    }

    Vector2Norm(rho_error, DMnd, &norm_error, comm);
    Vector2Norm(pSPARC->rho_data_inverse, DMnd, &norm_rho_data, comm);

    relative_norm_error = norm_error/norm_rho_data;

    if(!rank){
        printf("relative error in density: %.6E\n",relative_norm_error);
    }
    free(rho_error);

    int DMnd_procs[nproc];
    MPI_Allgather(&DMnd, 1, MPI_INT, DMnd_procs, 1, MPI_INT, MPI_COMM_WORLD);

    int rank_last_proc;

    for (int i = 0; i < nproc; i++){
        if (DMnd_procs[nproc-1-i] > 0){
            rank_last_proc = nproc - 1-i;
            break;
        }
    }
    pSPARC->rank_last_proc_inverse = rank_last_proc;



    if (rank != pSPARC->rank_last_proc_inverse){
        b = (double *) malloc(sizeof(double)*(DMnd));
        x0_linearsolve = (double *)calloc((DMnd), sizeof(double));
        x = (double *)calloc((DMnd), sizeof(double));
    } else {
        b = (double *) malloc(sizeof(double)*(DMnd+1));
        x0_linearsolve = (double *)calloc((DMnd+1), sizeof(double));
        x = (double *)calloc((DMnd+1), sizeof(double));
    }
	



	for (int i = 0; i < DMnd; i++){
		b[i] = -4*(eigvec_min[i]*eigvec_min[i] - pSPARC->rho_data_inverse[i])*eigvec_min[i];
	}

    if (rank==pSPARC->rank_last_proc_inverse){
	   b[DMnd] = 0.0;
    }

    



	if (pSPARC->count_invert_derivative > 0){
		for (int i = 0; i < DMnd; i++){
			x0_linearsolve[i] = pSPARC->x_guess_inverse[i];
		}
        if (rank == pSPARC->rank_last_proc_inverse){
            x0_linearsolve[DMnd] = pSPARC->x_guess_inverse[DMnd];
        }
	} else {
        if (rank == pSPARC->rank_last_proc_inverse){
		  pSPARC->x_guess_inverse = (double *) calloc((DMnd+1), sizeof(double));
        } else {
            pSPARC->x_guess_inverse = (double *) calloc((DMnd), sizeof(double));
        }
        for (int i = 0; i < DMnd; i++){
            x0_linearsolve[i] = pSPARC->x_guess_inverse[i];
        }

        if (rank == pSPARC->rank_last_proc_inverse){
            x0_linearsolve[DMnd] = pSPARC->x_guess_inverse[DMnd];
        }
	}
	


	CG_inverse(pSPARC, DMnd,  DMVertices, eigmin, eigvec_min,
        x0_linearsolve, b, x, pSPARC->MAXITER_CG_inverse, pSPARC->TOL_CG_inverse, comm);

    



	for (int i = 0; i < DMnd; i++){
		pSPARC->x_guess_inverse[i] = x[i];
	}

    if (rank == pSPARC->rank_last_proc_inverse){
        pSPARC->x_guess_inverse[DMnd] = x[DMnd];
    }

	for (int i = 0; i < DMnd; i++){
		pSPARC->dL_dVp[i] = -1.0*x[i] * eigvec_min[i];
	}




    pSPARC->count_invert_derivative++;

	free(eigvec_min);
	free(x0);
	free(b);
	free(x0_linearsolve);
	free(x);


}

/**
 * @brief   Conjugate Gradient (CG) method for solving a general linear system Ax = b. 
 *
 *          CG() assumes that x and  b is distributed among the given communicator. 
 *          Ax is calculated by calling function Ax().
 */
void CG_inverse(SPARC_OBJ *pSPARC, 
    int DMnd, int *DMVertices, double epsilon, double *u, double *x0, double *b,
     double *x, int max_iter, double tol,  MPI_Comm comm)
{   

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // FILE *fp;

    // if (!rank){
    //     fp = fopen("yy.txt","w");
    //     for (int i = 0; i < DMnd+1; i++){
    //         fprintf(fp,"%.14f\n",b[i]);
    //     }
    //     fclose(fp);
    // }
    // MPI_Barrier(MPI_COMM_WORLD);

    // exit(12);
    


    int i, j, iter_count = 0;
    double *r, *d, *q, *Ax, delta_new, delta_old, alpha, beta, err, b_2norm, delta0;

    if (rank == pSPARC->rank_last_proc_inverse){
        r = (double *)calloc( DMnd+1 , sizeof(double) );
        d = (double *)calloc( DMnd+1 , sizeof(double) );
        q = (double *)calloc( DMnd+1 , sizeof(double) );
        Ax = (double *)calloc(DMnd+1 , sizeof(double) ); 
    } else {
        r = (double *)calloc( DMnd , sizeof(double) );
        d = (double *)calloc( DMnd , sizeof(double) );
        q = (double *)calloc( DMnd , sizeof(double) );
        Ax = (double *)calloc(DMnd , sizeof(double) ); 
    }
       

    /********************************************************************/

    
    for (int i = 0; i < DMnd; i++){
    	x[i] =x0[i];
    }

    if (rank == pSPARC->rank_last_proc_inverse){
        x[DMnd] =x0[DMnd];
    }


    linear_system_Ax_inverse(pSPARC, DMnd, DMVertices, epsilon, u,
        x, Ax, comm);

    // FILE *fp;

    

    
    for (i = 0; i < DMnd; ++i){
        r[i] = b[i] - Ax[i];
        d[i] = r[i];
    }



    if(rank == pSPARC->rank_last_proc_inverse){
        r[DMnd] = b[DMnd] - Ax[DMnd];
        d[DMnd] = r[DMnd];
    }

    // if (!rank){
    //     fp = fopen("yy.txt","w");
    //     for (int i = 0; i < DMnd+1; i++){
    //         fprintf(fp,"%.14f\n",u[i]);
    //     }
    //     fclose(fp);
    // }
    // MPI_Barrier(MPI_COMM_WORLD);
    // exit(11);


    if (rank == pSPARC->rank_last_proc_inverse){
        Vector2Norm(r, DMnd+1, &delta_new, comm);
    } else {
        Vector2Norm(r, DMnd, &delta_new, comm);
    }
    
    delta_new  = delta_new*delta_new;
    delta0 = delta_new;

    err = tol + 1.0;
    while(iter_count < max_iter && delta_new > tol*tol*delta0){
    	linear_system_Ax_inverse(pSPARC, DMnd, DMVertices, epsilon, u,
        d, q, comm);    
        MPI_Barrier(MPI_COMM_WORLD);  

        // if (iter_count ==1){
        //     exit(124);
        // } 

        // if (iter_count==1){
        //     if (rank==1){
        //         fp = fopen("yy.txt","w");
        //         for (int i = 0; i < DMnd+1; i++){
        //             fprintf(fp,"%.14f %.14f\n",d[i], q[i]);
        //         }
        //         fclose(fp);
        //     }
        //     MPI_Barrier(MPI_COMM_WORLD);
        //     exit(15);
        // }
        
        

        // Ax(pSPARC, DMnd, DMVertices, 1, 0.0, d, q, comm);
        if (rank == pSPARC->rank_last_proc_inverse){
            VectorDotProduct(d, q, DMnd+1, &alpha, comm);
        } else {
            VectorDotProduct(d, q, DMnd, &alpha, comm);
        }
        

        alpha =  delta_new / alpha;
        
        



        for (j = 0; j < DMnd; j++){
            x[j] = x[j] + alpha * d[j];
        }

        if (rank == pSPARC->rank_last_proc_inverse){
            x[DMnd] = x[DMnd] + alpha * d[DMnd];
        }



        if ((iter_count)%50 ==0){
            linear_system_Ax_inverse(pSPARC, DMnd, DMVertices, epsilon, u,
                x, Ax, comm);
            for (j = 0; j < DMnd; j++){    
                r[j] = b[j] -  Ax[j];
            }
            if (rank == pSPARC->rank_last_proc_inverse){
                r[DMnd] = b[DMnd] -  Ax[DMnd];
            }
        } else {
            for (j = 0; j < DMnd; j++){
                r[j] = r[j] - alpha * q[j];
            }
            if (rank == pSPARC->rank_last_proc_inverse){
                r[DMnd] = r[DMnd] - alpha * q[DMnd];
            }
        }


        delta_old = delta_new;

        if (rank ==pSPARC->rank_last_proc_inverse){
            Vector2Norm(r, DMnd+1, &delta_new, comm);
        } else {
            Vector2Norm(r, DMnd, &delta_new, comm);
        }
        
        delta_new =  delta_new*delta_new;


        beta = delta_new/delta_old;

        // if (!rank){
        //     fp = fopen("yy.txt","a");
        //     fprintf(fp,"%.14f %.14f %.14f\n",alpha, delta_new, beta);
        //     fclose(fp);
        // }
       

        for (int ii = 0; ii < DMnd; ii++){
        	d[ii] = r[ii] + beta * d[ii];
        }
        if (rank == pSPARC->rank_last_proc_inverse){
            d[DMnd] = r[DMnd] + beta * d[DMnd];
        }

        // if (rank){
        //     fp = fopen("yy.txt","a");
        //     for (int jj = 0; jj < DMnd+1; jj++){
        //         fprintf(fp,"%.14f\n",d[jj]);
        //     }
        //     fclose(fp);
        // }
        // MPI_Barrier(MPI_COMM_WORLD);
        // exit(13);


        iter_count=iter_count+1;
    }
    if (!rank){
        if (fabs(sqrt(delta_new)) > tol) {
            printf("WARNING: CG only converged to %.5E!\n", sqrt(delta_new));
        } else {
            printf("CG converged to %.5E in %d iteration\n",sqrt(delta_new), iter_count);
        }
    }

    // MPI_Barrier(MPI_COMM_WORLD);
    // exit(7);
    


    free(r);
    free(d);
    free(q);
    free(Ax);
}




/*
@brief: function to perform Limited memory version of BFGS for structural relaxation (based on the implementation in VTST)
*/
void LBFGS_inverse(SPARC_OBJ *pSPARC) {
    double t_init, t_acc;
    t_init = MPI_Wtime();
    int rank, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

#ifdef DEBUG
    if (rank == 0){
        printf(GRN "Starting L-BFGS for OFDFT inversion\n"RESET);
    }
#endif

    double lbfgs_tol = pSPARC->BFGS_TOL_inverse;
    // int n = 3 * pSPARC->n_atom;
    int n =  pSPARC->Nd_d;

    int m = pSPARC->BFGS_history_inverse;
    double finit_stp = pSPARC->L_finit_stp;
    double maxmov = pSPARC->L_maxmov;
    int autoscale = pSPARC->L_autoscale;
    int lineopt = pSPARC->L_lineopt; // Needed only if autoscale = 0
    double icurv = pSPARC->L_icurv; // Needed only if autoscale = 0

    double damp = 2.0;
    pSPARC->isFD = 1; // Never change here
    pSPARC->isReset = 1; // Never change here
    pSPARC->step = 0; // Never change here

    double *alpha, *xold; // Init_pos
    double err_collect[nproc];
    double temp_val=0.0;
    pSPARC->deltaX = (double *)calloc( m*n , sizeof(double) );
    pSPARC->deltaG = (double *)calloc( m*n , sizeof(double) );
    pSPARC->iys = (double *) calloc(m , sizeof(double));
    alpha = (double *) malloc(m * sizeof(double));
    xold = (double *) malloc(n * sizeof(double));
    pSPARC->fold = (double *) malloc(n * sizeof(double));
    pSPARC->d = (double *) malloc(n * sizeof(double));
    pSPARC->atom_disp = (double *) malloc(n * sizeof(double));

    MPI_Comm comm;
    comm = pSPARC->dmcomm_phi;

    int i, j, k, maxmov_flag, s_pos, bound;
    double curv, fnorm, dnorm, fp1, fp2, stp_sz, favg, beta, a1, a2;

    int check = (pSPARC->PrintRelaxout == 1 && !rank), check1 = (pSPARC->Printrestart == 1 && !rank);
    int iter;
    double err;
    double Vp_shift;
    int DMnd = pSPARC->Nd_d;

    // // Check whether the restart has to be performed
    // if(pSPARC->RestartFlag != 0){
    //     RestartRelax(pSPARC); // collects atomic positions
    //     int atm;
    //     if(pSPARC->cell_typ != 0){
    //         for(atm = 0; atm < pSPARC->n_atom; atm++){
    //             Cart2nonCart_coord(pSPARC, &pSPARC->atom_pos[3*atm], &pSPARC->atom_pos[3*atm+1], &pSPARC->atom_pos[3*atm+2]);
    //         }
    //     }
    //     Calculate_electronicGroundState(pSPARC);
    //     err = 0.0;
    //     for(i = 0; i < n; i++){
    //         if (fabs(pSPARC->forces[i]) > err)
    //             err = fabs(pSPARC->forces[i]); // defined as supremum norm of force vector
    //     }
    // } else {
    //     Calculate_electronicGroundState(pSPARC);
    //     err = 0.0;
    //     for(i = 0; i < n; i++){
    //         if (fabs(pSPARC->forces[i]) > err)
    //             err = fabs(pSPARC->forces[i]); // defined as supremum norm of force vector
    //     }
    // }


    Calculate_Inversion_derivative(pSPARC);
    err = lbfgs_tol+1.0;
    for (int i = 0; i < n; i++){
    	if (fabs(pSPARC->dL_dVp[i]) > err){
    		err = fabs(pSPARC->dL_dVp[i]); // defined as supremum norm of force vector
    	}
    }



    pSPARC->elecgs_Count++;
    pSPARC->RelaxCount++;

    int imax = pSPARC->BFGS_MAXITER_inverse;
    FILE *output_relax, *output_fp;
    // if(check){
    //     output_relax = fopen(pSPARC->RelaxFilename,"a");
    //     if(output_relax == NULL){
    //         printf("\nCannot open file \"%s\"\n",pSPARC->RelaxFilename);
    //         exit(EXIT_FAILURE);
    //     }
        
    //     output_fp = fopen(pSPARC->OutFilename,"a");
    //     if (output_fp == NULL) {
    //         printf("\nCannot open file \"%s\"\n",pSPARC->OutFilename);
    //         exit(EXIT_FAILURE);
    //     }
    //     fprintf(output_fp,"Relax step time                    :  %.3f (sec)\n", (MPI_Wtime() - t_init));
    //     fclose(output_fp);

    //     if(pSPARC->RestartFlag == 0){
    //         fprintf(output_relax,":RELAXSTEP: %d\n", pSPARC->RelaxCount);
    //         Print_fullRelax(pSPARC, output_relax); // prints the QOI in the output_relax file
    //     }
    //     fclose(output_relax);

    // }

    iter = pSPARC->RelaxCount + pSPARC->restartCount + (pSPARC->RestartFlag == 0);
    t_acc = (MPI_Wtime() - t_init)/60.0;

    int gridsizes[3];
    gridsizes[0] = pSPARC->Nx;
    gridsizes[1] = pSPARC->Ny;
    gridsizes[2] = pSPARC->Nz;
    int *DMVertices;
    DMVertices = pSPARC->DMVertices;
    char fname[] = "Vp.cube";

    // TODO: Compute Hessian matrix directly later using perturbation theory
    while (iter < imax && err > lbfgs_tol && (t_acc + 1.2 * (MPI_Wtime() - t_init)/60.0) < pSPARC->TWtime) {
        t_init = MPI_Wtime();
        // if (check){
        //     output_relax = fopen(pSPARC->RelaxFilename,"a+");
        //     if (output_relax == NULL) {
        //         printf("\nCannot open file \"%s\"\n",pSPARC->RelaxFilename);
        //         exit(EXIT_FAILURE);
        //     }
        //     fprintf(output_relax,":RELAXSTEP: %d\n", iter);
        // }

#ifdef DEBUG
        if(!rank)
            printf(":RelaxStep: %d\n",iter);
#endif
        maxmov_flag = 0;
        if(autoscale){
            if(pSPARC->isFD){
            // Take the finite difference pSPARC->step down the forces

                // fnorm = norm(n, pSPARC->dL_dVp);
                Vector2Norm(pSPARC->dL_dVp, n, &fnorm, comm);  // changed here
                for(i = 0; i < n; i++){
                    pSPARC->d[i] = pSPARC->dL_dVp[i]/fnorm; // normalized force direction
                    xold[i] = pSPARC->Vp_inverse[i];
                    pSPARC->fold[i] = pSPARC->dL_dVp[i];
                    pSPARC->Vp_inverse[i] += pSPARC->d[i] * finit_stp; // finite difference pSPARC->step in the direction of the force
                }
                pSPARC->Relax_fac = finit_stp; // Needed for charge extrapolation
                pSPARC->isFD = 0;
            } else{
                for(i = 0; i < n; i++){
                    pSPARC->d[i] = pSPARC->atom_disp[i];
                }
                // dnorm = norm(n, pSPARC->d);
                Vector2Norm(pSPARC->d, n, &dnorm, comm);  // changed here
                // fp1 = dotproduct(n, pSPARC->fold, 0, pSPARC->d, 0)/dnorm;
                // fp2 = dotproduct(n, pSPARC->dL_dVp, 0, pSPARC->d, 0)/dnorm;
                VectorDotProduct(pSPARC->fold, pSPARC->d, n, &fp1, comm);  // changed here
                VectorDotProduct(pSPARC->dL_dVp, pSPARC->d, n, &fp2, comm);  // changed here
                fp1 = fp1/dnorm;  // changed here
                fp2 = fp2/dnorm;  // changed here

                curv = (fp1 - fp2)/dnorm;
                icurv = 1.0/(curv * damp); // To be used as guess for inverse Hessian
                if(icurv < 0){
                    pSPARC->isReset = 1;
                    maxmov_flag = 1;
                }
                if(pSPARC->isReset == 1){
                    pSPARC->step = 0;
                    pSPARC->isReset = 0;
                } else{
                    if(pSPARC->step < m){
                        s_pos = pSPARC->step * n;
                        for(i = 0; i < n; i++){
                            pSPARC->deltaX[s_pos + i] = pSPARC->atom_disp[i];
                            pSPARC->deltaG[s_pos + i] = pSPARC->fold[i] - pSPARC->dL_dVp[i];
                        }
                        // pSPARC->iys[pSPARC->step] = 1.0/dotproduct(n, pSPARC->deltaX, s_pos, pSPARC->deltaG, s_pos);
                        VectorDotProduct(&pSPARC->deltaX[s_pos], &pSPARC->deltaG[s_pos], n, &temp_val, comm); // changed here
                        pSPARC->iys[pSPARC->step] = 1.0/temp_val; // changed here
                    } else{
                        s_pos = (m-1) * n;
                        for(i = 0; i < s_pos; i++){
                            pSPARC->deltaX[i] = pSPARC->deltaX[i + n];
                            pSPARC->deltaG[i] = pSPARC->deltaG[i + n];
                        }
                        for(i = 0; i < m-1; i++)
                            pSPARC->iys[i] = pSPARC->iys[i + 1];
                        for(i = 0; i < n; i++){
                            pSPARC->deltaX[s_pos + i] = pSPARC->atom_disp[i];
                            pSPARC->deltaG[s_pos + i] = pSPARC->fold[i] - pSPARC->dL_dVp[i];
                        }
                        // pSPARC->iys[m-1] = 1.0/dotproduct(n, pSPARC->deltaX, s_pos, pSPARC->deltaG, s_pos);
                        VectorDotProduct(&pSPARC->deltaX[s_pos], &pSPARC->deltaG[s_pos], n, &temp_val, comm); // changed here
                        pSPARC->iys[m-1] = 1.0/temp_val; // changed here
                    }
                    pSPARC->step++;
                }
                for(i = 0; i < n; i++){
                    xold[i] = pSPARC->Vp_inverse[i];
                    pSPARC->fold[i] = pSPARC->dL_dVp[i];
                }
                if(pSPARC->step < m)
                    bound = pSPARC->step;
                else
                    bound = m;

                // Perform rank two BFGS update
                for(i = 0; i < n; i++)
                    pSPARC->d[i] = -pSPARC->dL_dVp[i];

                for(i = 0; i < bound; i++){
                    j = bound - i - 1;
                    s_pos = j * n;
                    // alpha[j] = dotproduct(n, pSPARC->deltaX, s_pos, pSPARC->d, 0);
                    VectorDotProduct(&pSPARC->deltaX[s_pos], &pSPARC->d[0], n, &temp_val, comm); // changed here
                    alpha[j] = temp_val; // changed here
                    alpha[j] *= pSPARC->iys[j];
                    for(k = 0; k < n; k++)
                        pSPARC->d[k] -= alpha[j] * pSPARC->deltaG[s_pos + k];
                }
                for(i = 0; i < n; i++)
                    pSPARC->d[i] = icurv * pSPARC->d[i];
                for(i = 0; i < bound; i++){
                    s_pos = i * n;
                    // beta = dotproduct(n, pSPARC->deltaG, s_pos, pSPARC->d, 0);
                    VectorDotProduct(&pSPARC->deltaG[s_pos], &pSPARC->d[0], n, &beta, comm); // changed here
                    beta *= pSPARC->iys[i];
                    for(k = 0; k < n; k++)
                        pSPARC->d[k] += pSPARC->deltaX[s_pos + k] * (alpha[i] - beta);
                }
                for(i = 0; i < n; i++){
                    pSPARC->d[i] = -pSPARC->d[i];
                }

                // stp_sz = norm(n, pSPARC->d);
                Vector2Norm(pSPARC->d, n, &stp_sz, comm);  // changed here

                if(stp_sz > maxmov){
                    pSPARC->isReset = 1;
                    stp_sz = maxmov;
                    // fnorm = norm(n, pSPARC->dL_dVp);
                    Vector2Norm(pSPARC->dL_dVp, n, &fnorm, comm);  // changed here
                    for(i = 0; i < n; i++)
                        pSPARC->d[i] = stp_sz * pSPARC->dL_dVp[i]/fnorm; //  Take a steepest descent pSPARC->step
                }
                if(maxmov_flag){
                    // fnorm = norm(n, pSPARC->dL_dVp);
                    Vector2Norm(pSPARC->dL_dVp, n, &fnorm, comm);  // changed here
                    for(i = 0; i < n; i++){
                        pSPARC->d[i] = pSPARC->dL_dVp[i]/fnorm;
                        pSPARC->Vp_inverse[i] += maxmov * pSPARC->d[i];
                    }
                    pSPARC->Relax_fac = maxmov;
                    maxmov_flag = 0;
                } else{
                    for(i = 0; i < n; i++)
                        pSPARC->Vp_inverse[i] += pSPARC->d[i];
                    pSPARC->Relax_fac = 1.0;
                }
            }
        } else{
            if(pSPARC->isFD){
                pSPARC->isFD = 0;
                // a1 = fabs(dotproduct(n, pSPARC->dL_dVp, 0, pSPARC->fold, 0));
                // a2 = dotproduct(n, pSPARC->fold, 0, pSPARC->fold, 0);

                VectorDotProduct(pSPARC->dL_dVp, pSPARC->fold, n, &temp_val, comm); // changed here
                a1 = fabs(temp_val); // changed here
                VectorDotProduct(pSPARC->fold, pSPARC->fold, n, &a2, comm); // changed here


                if(a1 > 0.5 * a2 || a2 == 0)
                    pSPARC->isReset = 1;
                if(lineopt == 0)
                    pSPARC->isReset = 0;
                if(a2 == 0)
                    pSPARC->isReset = 1;
                if(pSPARC->isReset){
                    pSPARC->step = 0;
                    pSPARC->isReset = 0;
                } else{
                    if(pSPARC->step < m){
                        s_pos = pSPARC->step * n;
                        for(i = 0; i < n; i++){
                            pSPARC->deltaX[s_pos + i] = pSPARC->atom_disp[i];
                            pSPARC->deltaG[s_pos + i] = pSPARC->fold[i] - pSPARC->dL_dVp[i];
                        }
                        // pSPARC->iys[pSPARC->step] = 1.0/dotproduct(n, pSPARC->deltaX, s_pos, pSPARC->deltaG, s_pos);
                        VectorDotProduct(&pSPARC->deltaX[s_pos], &pSPARC->deltaG[s_pos], n, &temp_val, comm); // changed here
                        pSPARC->iys[pSPARC->step] = 1.0/temp_val; // changed here
                    } else{
                        s_pos = (m-1) * n;
                        for(i = 0; i < s_pos; i++){
                            pSPARC->deltaX[i] = pSPARC->deltaX[i + n];
                            pSPARC->deltaG[i] = pSPARC->deltaG[i + n];
                        }
                        for(i = 0; i < m-1; i++)
                            pSPARC->iys[i] = pSPARC->iys[i + 1];
                        for(i = 0; i < n; i++){
                            pSPARC->deltaX[s_pos + i] = pSPARC->atom_disp[i];
                            pSPARC->deltaG[s_pos + i] = pSPARC->fold[i] - pSPARC->dL_dVp[i];
                        }
                        // pSPARC->iys[m-1] = 1.0/dotproduct(n, pSPARC->deltaX, s_pos, pSPARC->deltaG, s_pos);
                        VectorDotProduct(&pSPARC->deltaX[s_pos], &pSPARC->deltaG[s_pos], n, &temp_val, comm); // changed here
                        pSPARC->iys[m-1] = 1.0/temp_val;  // changed here
                    }
                    pSPARC->step++;
                }
                for(i = 0; i < n; i++){
                    xold[i] = pSPARC->Vp_inverse[i];
                    pSPARC->fold[i] = pSPARC->dL_dVp[i];
                }
                if(pSPARC->step < m)
                    bound = pSPARC->step;
                else
                    bound = m;

                // Perform rank 2 BFGS update

                for(i = 0; i < n; i++)
                    pSPARC->d[i] = -pSPARC->dL_dVp[i];
                for(i = 0; i < bound; i++){
                    j = bound - i - 1;
                    s_pos = j * n;
                    // alpha[j] = dotproduct(n, pSPARC->deltaX, s_pos, pSPARC->d, 0);
                    VectorDotProduct(&pSPARC->deltaX[s_pos], pSPARC->d, n, &temp_val, comm); // changed here
                    alpha[j] = temp_val; // changed here
                    alpha[j] *= pSPARC->iys[j];
                    for(k = 0; k < n; k++)
                        pSPARC->d[k] -= alpha[j] * pSPARC->deltaG[s_pos + k];
                }
                for(i = 0; i < n; i++)
                    pSPARC->d[i] = icurv * pSPARC->d[i];
                for(i = 0; i < bound; i++){
                    s_pos = i * n;
                    // beta = dotproduct(n, pSPARC->deltaG, s_pos, pSPARC->d, 0);
                    VectorDotProduct(&pSPARC->deltaG[s_pos], pSPARC->d, n, &temp_val, comm); // changed here
                    beta = temp_val; // changed here
                    beta *= pSPARC->iys[i];
                    for(k = 0; k < n; k++)
                        pSPARC->d[k] += pSPARC->deltaX[s_pos + k] * (alpha[i] - beta);
                }
                for(i = 0; i < n; i++)
                    pSPARC->d[i] = -pSPARC->d[i];

                if(lineopt){
                    // dnorm = norm(n, pSPARC->d);
                    Vector2Norm(pSPARC->d, n, &dnorm, comm);  // changed here
                    for(i = 0; i < n; i++){
                        pSPARC->d[i] /= dnorm;
                        pSPARC->Vp_inverse[i] += pSPARC->d[i] * finit_stp; // finite difference pSPARC->step along search direction
                    }
                    pSPARC->Relax_fac = finit_stp;
                } else{
                    // stp_sz = dnorm = norm(n, pSPARC->d);
                    Vector2Norm(pSPARC->d, n, &dnorm, comm);  // changed here
                    stp_sz = dnorm; // changed here
                    if(stp_sz > maxmov){
                        stp_sz = maxmov;
                        for(i = 0; i < n; i++)
                            pSPARC->d[i] = stp_sz * pSPARC->d[i]/dnorm;
                    }
                    for(i = 0; i < n; i++)
                        pSPARC->Vp_inverse[i] += pSPARC->d[i];
                    pSPARC->Relax_fac = 1.0;
                    pSPARC->isFD = 1;
                }
            } else{
                pSPARC->isFD = 1;
                // fp1 = dotproduct(n, pSPARC->fold, 0, pSPARC->d, 0);
                // fp2 = dotproduct(n, pSPARC->dL_dVp, 0, pSPARC->d, 0);
                VectorDotProduct(pSPARC->fold, pSPARC->d, n, &fp1, comm); // changed here
                VectorDotProduct(pSPARC->dL_dVp, pSPARC->d, n, &fp2, comm); // changed here

                curv = (fp1 - fp2)/finit_stp;
                if(curv < 0)
                    stp_sz = maxmov;
                else{
                    favg = 0.5 * (fp1 + fp2);
                    stp_sz = favg/curv;
                    if(fabs(stp_sz) > maxmov)
                        stp_sz = SIGN(maxmov, stp_sz) - SIGN(finit_stp, stp_sz);
                    else
                        stp_sz -= 0.5 * finit_stp;
                }
                for(i = 0; i < n; i++)
                    pSPARC->Vp_inverse[i] += pSPARC->d[i] * stp_sz;
                pSPARC->Relax_fac = stp_sz;
            }
        }

        // Store the distance the atoms have moved between two relaxation iterations
        for(i = 0; i < n; i++)
            pSPARC->atom_disp[i] = pSPARC->Vp_inverse[i] - xold[i];

        // elecDensExtrapolation(pSPARC);
        // Check_atomlocation(pSPARC);
        // Calculate_electronicGroundState(pSPARC);

        Vp_shift = 0.0;
        VectorSum(pSPARC->Vp_inverse, DMnd, &Vp_shift, pSPARC->dmcomm_phi);
        Vp_shift /= (double)pSPARC->Nd;
        VectorShift(pSPARC->Vp_inverse, DMnd, -Vp_shift, pSPARC->dmcomm_phi);

        Calculate_Inversion_derivative(pSPARC);
        pSPARC->elecgs_Count++;
        err = 0.0;
        for(i = 0; i < n; i++){
            if (fabs(pSPARC->dL_dVp[i]) > err)
                err = fabs(pSPARC->dL_dVp[i]); // defined as supremum norm of force vector
        }
        MPI_Allgather(&err, 1, MPI_DOUBLE, err_collect, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        err = 0.0;
        for (int ii = 0; ii < nproc; ii++){
            if (fabs(err_collect[ii])>err){
                err = fabs(err_collect[ii]);
            }
        }

        // if(check){
        //     Print_fullRelax(pSPARC, output_relax); // prints the QOI in the output_relax file
        //     fclose(output_relax);
        // }
        // if(check1 && !(iter % pSPARC->Printrestart_fq)) // printrestart_fq is the frequency at which the restart file is written
        //     PrintRelax(pSPARC);
        if(access("SPARC.stop", F_OK ) != -1 ){ // If a .stop file exists in the folder then the run will be terminated
            pSPARC->RelaxCount++;
            break;
        }
#ifdef DEBUG
        if (!rank) printf("Time taken by RelaxStep %d: %.3f s.\n", iter, (MPI_Wtime() - t_init));
#endif
        if(!rank){
            output_fp = fopen(pSPARC->OutFilename,"a");
            if (output_fp == NULL) {
                printf("\nCannot open file \"%s\"\n",pSPARC->OutFilename);
                exit(EXIT_FAILURE);
            }
            fprintf(output_fp,"Relax step time                    :  %.3f (sec)\n", (MPI_Wtime() - t_init));
            fclose(output_fp);
        }

        if (iter %100 ==0){
            print_vec(pSPARC->Vp_inverse, gridsizes, DMVertices, fname, comm);
        }

        pSPARC->RelaxCount++;
        iter++;
        t_acc += (MPI_Wtime() - t_init)/60.0;

    }



    if(check1){
        pSPARC->RelaxCount--;
        PrintRelax(pSPARC);
    }
    free(pSPARC->deltaX);
    free(pSPARC->deltaG);
    free(pSPARC->iys);
    free(alpha);
    free(xold);
    free(pSPARC->fold);
    free(pSPARC->atom_disp);
    free(pSPARC->d);
}