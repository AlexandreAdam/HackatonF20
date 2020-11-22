#!/bin/bash
#SBATCH --account=def-lplevass
#SBATCH --gres=gpu:2              # Number of GPU(s) per node
#SBATCH --cpus-per-task=6         # CPU cores/threads
#SBATCH --mem=4000M               # memory per node
#SBATCH --time=0-03:00            # time (DD-HH:MM)
module load python/3.6
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index -r requirement_beluga.txt
python bert_features.py
