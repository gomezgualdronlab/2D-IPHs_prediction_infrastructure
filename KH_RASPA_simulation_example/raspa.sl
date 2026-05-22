#!/bin/bash -x 

#SBATCH --job-name="dmfHK-SR_fmj_v1-3c_B_Ch_v2-4c_Cu_1_Ch_v3-4c_Cu_1_Ch_v4-3c_B_Ch_v5-4c_Cu_1_Ch_2-ntn_edge_2-1B_2NO2_Chnl_1x1x1"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --export=ALL
#SBATCH --time=5-00:00:00

cd $SLURM_SUBMIT_DIR

EXE=/RASPA/bin/simulate

srun -n 1 $EXE simulation.input

