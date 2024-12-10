#!/bin/bash
#SBATCH -JAl_DFT
#SBATCH --account=hive-phanish6
#SBATCH -phive-interact
#SBATCH -N 1 --ntasks-per-node=24
#SBATCH --mem-per-cpu=7G
#SBATCH -t1:00:00
cd $SLURM_SUBMIT_DIR

module load intel

echo $PWD 

srun ./sparc -name H > log.txt
