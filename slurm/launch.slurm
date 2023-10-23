#!/bin/bash
#SBATCH -J sut
#SBATCH -N 16
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem=64gb

echo $(pwd)
srun bash -c "TIMESTAMP=$TIMESTAMP EXP_NAME=$EXP_NAME slurm/distributed_run.sh"
