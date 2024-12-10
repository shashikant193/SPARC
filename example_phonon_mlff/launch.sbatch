#!/bin/bash
#SBATCH -JMLFF_hugoniot
#SBATCH --account=gts-phanish6-coda20
#SBATCH -p inferno
#SBATCH -N 1 --ntasks-per-node=2
#SBATCH --mem-per-cpu=20G
#SBATCH -t03:30:00
cd $SLURM_SUBMIT_DIR
module load intel
echo $PWD 
srun /storage/scratch1/5/skumar416/JMPS_run/SOURCE_CODE/PHONON/lib/sparc -name C > log.txt