#!/bin/bash
#SBATCH -JMLFF_hugoniot
#SBATCH --account=gts-phanish6-coda20
#SBATCH -p inferno
#SBATCH -N 5 --ntasks-per-node=24
#SBATCH --mem-per-cpu=7G
#SBATCH -t1:00:00
cd $SLURM_SUBMIT_DIR
module load intel
echo $PWD 
srun ../lib_retrain/sparc -name C > C_log.txt
