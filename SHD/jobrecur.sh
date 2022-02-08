#!/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --mem 20G
#SBATCH -t 1-00:00
#SBATCH --job-name=RSNN
#SBATCH -o RSNN128oglr/RSNN128lrrun1NEW2.out
#SBACTH -e slurm.%j.err
#SBATCH --mail-user=f20180790@hyderabad.bits-pilani.ac.in
#SBATCH --mail-type=ALL
module load cuda-11.0.2-gcc-10.2.0-3wlbq6u
srun python3 testrecurrent128.py

