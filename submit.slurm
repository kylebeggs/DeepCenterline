#!/bin/bash

#SBATCH --gpus-per-node=2
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=4
#SBATCH --gres-flags=enforce-binding
#SBATCH --mem-per-gpu=10G
#SBATCH --time=12:00:00

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user kbeggs07@knights.ucf.edu

# Load modules
module load anaconda/anaconda3
module list
source activate torch-medical

# check gpu allocation
nvidia-smi
nvidia-smi topo -m

# run model
cd src
python train.py --name pooling --model pooling --epochs 20 --lr 0.01
