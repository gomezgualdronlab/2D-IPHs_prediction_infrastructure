#!/bin/bash -x 

#SBATCH --job-name="MOF_name"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --export=ALL
#SBATCH --time=5-00:00:00

cd $SLURM_SUBMIT_DIR

EXE=/RASPA/bin/simulate

srun -n 1 $EXE simulation.input

