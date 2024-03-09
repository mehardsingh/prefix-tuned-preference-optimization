#!/bin/bash

#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=128GB
#SBATCH --account=mihalcea0
#SBATCH --time=00-08:00:00 

# set up job
echo ====Setting up environment====
module load cuda
eval "$(conda shell.bash hook)"
conda activate ptpo2
echo ====Environment setup finished====

pushd /home/mehars/prefix-tuned-preference-optimization

# run job

python country_prefix/src/encoder_rm_all/train_encoder_rm_all.py --data_dir country_prefix/data --model_name distilbert-base-uncased --batch_size 16 --wd 1e-2 --lr 2e-5 --num_epochs 3 --save_dir country_prefix/train_status/encoder_rm_all

# python country_prefix/src/encoder_rm_single/train_encoder_rm_single.py --country Belgium --data_dir country_prefix/data --model_name distilbert-base-uncased --batch_size 16 --wd 1e-2 --lr 2e-5 --num_epochs 3 --save_dir country_prefix/train_status/encoder_rm_single
# python country_prefix/src/encoder_rm_single/train_encoder_rm_single.py --country "United States" --data_dir country_prefix/data --model_name distilbert-base-uncased --batch_size 16 --wd 1e-2 --lr 2e-5 --num_epochs 3 --save_dir country_prefix/train_status/encoder_rm_single