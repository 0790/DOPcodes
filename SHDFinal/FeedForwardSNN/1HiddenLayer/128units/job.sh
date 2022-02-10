#!/bin/bash
#SBATCH -p gpu_x2
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --mem 20G
#SBATCH -t 1-00:00
#SBATCH --job-name=SNN128
#SBATCH -o Outputs/TEST1RUN1.out
#SBACTH -e slurm.%j.err
#SBATCH --mail-user=f20180790@hyderabad.bits-pilani.ac.in
#SBATCH --mail-type=ALL
module load nvhpc-22.1-gcc-8.5.0-sdi5tb5
srun python3 testff128.py
