#!/bin/bash
#SBATCH -p gpu_x2
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --mem 30G
#SBATCH -t 1-00:00
#SBATCH --job-name=2562_0.00001S
#SBATCH -o SNN2layers256lr-0.00001/SNN2layers256lr-0.00001run3.out
#SBACTH -e slurm.%j.err
#SBATCH --mail-user=f20180790@hyderabad.bits-pilani.ac.in
#SBATCH --mail-type=ALL
module load cuda-11.0.2-gcc-10.2.0-3wlbq6u
srun python3 testff256_2layerslr-0.00001.py
