#!/bin/bash
#SBATCH --mem=64G
#SBATCH -p gpu
#SBATCH -c 2
#SBATCH --gpus-per-node=1
#SBATCH --gres=tmpspace:100G
#SBATCH -t 300:00:00 

source /hpc/dla_patho/premium/rens/miniconda3/etc/profile.d/conda.sh
conda activate hovernet

python /hpc/dla_patho/premium/rens/premium_pathology/hovernet_re/process_wsi.py