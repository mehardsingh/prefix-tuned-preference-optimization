echo ====Setting up environment====
module load python/3.9.12 cuda
source env/bin/activate
echo ====Environment setup finished====

pushd /home/mehars/prefix-tuned-preference-optimization

# run job

python country_prefix2/src/compute_alignment.py