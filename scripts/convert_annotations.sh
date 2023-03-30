#!/bin/bash

#SBATCH -t 100:00:00 

env
source /hpc/dla_patho/premium/rens/miniconda3/etc/profile.d/conda.sh
conda activate rens

export LD_LIBRARY_PATH=/hpc/local/CentOS7/dla_patho/packages/openslide/lib:$LD_LIBRARY_PATH


cd "/hpc/dla_patho/premium/rens/premium_pathology/scripts"
python convert_annotations.py
