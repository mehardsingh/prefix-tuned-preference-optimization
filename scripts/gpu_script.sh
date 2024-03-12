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

# python country_prefix/src/train_encoder_rm.py --country "United States" --data_dir country_prefix/data --model_name distilbert-base-uncased --batch_size 16 --wd 1e-2 --lr 2e-5 --num_epochs 10 --save_dir country_prefix/train_status/encoder_rm_all
# python country_prefix/src/train_encoder_rm.py --country "China" --data_dir country_prefix/data --model_name distilbert-base-uncased --batch_size 16 --wd 1e-2 --lr 2e-5 --num_epochs 10 --save_dir country_prefix/train_status/encoder_rm_all
# python country_prefix/src/train_encoder_rm.py --data_dir country_prefix/data --model_name distilbert-base-uncased --batch_size 16 --wd 1e-2 --lr 2e-5 --num_epochs 10 --save_dir country_prefix/train_status/encoder_rm_all

# python country_prefix/src/train_decoder_rm.py --data_dir country_prefix/data --model_name meta-llama/Llama-2-7b-hf --batch_size 16 --wd 1e-2 --lr 2e-5 --num_epochs 3 --save_dir country_prefix/train_status/decoder_rm
# python country_prefix/src/train_encoder2.py --data_dir country_prefix/data --model_name distilbert/distilroberta-base --batch_size 16 --wd 1e-2 --lr 2e-5 --num_epochs 3 --save_dir country_prefix/train_status/encoder_rm_all2

python country_prefix/src/train_encoder_rm.py --country "United States" --data_dir country_prefix/data --model_name distilbert/distilroberta-base --batch_size 16 --wd 1e-2 --lr 2e-5 --num_epochs 10 --save_dir country_prefix/train_status/encoder_rm
python country_prefix/src/train_encoder_rm.py --country "China" --data_dir country_prefix/data --model_name distilbert/distilroberta-base --batch_size 16 --wd 1e-2 --lr 2e-5 --num_epochs 10 --save_dir country_prefix/train_status/encoder_rm
python country_prefix/src/train_encoder_rm.py --data_dir country_prefix/data --model_name distilbert/distilroberta-base --batch_size 16 --wd 1e-2 --lr 2e-5 --num_epochs 10 --save_dir country_prefix/train_status/encoder_rm
