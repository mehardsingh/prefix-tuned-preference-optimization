#!/bin/bash

#SBATCH --mem-per-cpu=16GB
#SBATCH --time=00-08:00:00 

echo ====Setting up environment====
module load python/3.9.12 cuda
source env/bin/activate
echo ====Environment setup finished====

pushd /home/mehars/prefix-tuned-preference-optimization

# run job

python evaluate_attributes.py