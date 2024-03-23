#!/bin/bash

#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=128GB
#SBATCH --account=mihalcea0
#SBATCH --time=00-08:00:00 

# set up job
echo ====Setting up environment====
module load python/3.9.12 cuda
source env/bin/activate
echo ====Environment setup finished====

pushd /home/mehars/prefix-tuned-preference-optimization

# run job

# python country_prefix/src/train_roberta_rm.py --country_prefix "False" --lora "False" --country "United States" --data_dir country_prefix/data --model_name FacebookAI/roberta-base --batch_size 16 --wd 1e-2 --lr 2e-5 --num_epochs 20 --save_dir country_prefix2/checkpoints/roberta_rm
# python country_prefix/src/train_roberta_rm.py --country_prefix "False" --lora "False" --country "China" --data_dir country_prefix/data --model_name FacebookAI/roberta-base --batch_size 16 --wd 1e-2 --lr 2e-5 --num_epochs 20 --save_dir country_prefix2/checkpoints/roberta_rm
python country_prefix/src/train_roberta_rm.py --country_prefix "True" --lora "False" --data_dir country_prefix/data --model_name FacebookAI/roberta-base --batch_size 16 --wd 1e-2 --lr 2e-5 --num_epochs 3 --save_dir country_prefix2/checkpoints/roberta_rm
python country_prefix/src/train_roberta_rm.py --country_prefix "False" --lora "False" --data_dir country_prefix/data --model_name FacebookAI/roberta-base --batch_size 16 --wd 1e-2 --lr 2e-5 --num_epochs 3 --save_dir country_prefix2/checkpoints/roberta_rm
