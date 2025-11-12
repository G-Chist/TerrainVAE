#!/bin/bash
#SBATCH -N 1                      # allocate 1 compute node
#SBATCH -n 1                      # total number of tasks
#SBATCH --mem=8g                  # allocate 8 GB of memory
#SBATCH -J "TerrainVAE"      # name of the job
#SBATCH -o terrainvae_%j.out # name of the output file
#SBATCH -e terrainvae_%j.err # name of the error file
#SBATCH -p short                  # partition to submit to
#SBATCH -t 20:00:00               # time limit of 20 hours
#SBATCH --gres=gpu:1              # request 1 GPU

module load python/3.11.10

module load cuda/12.4.0

echo "Module loads done, running pip install"

pip install -r requirements.txt

echo "Pip installs done, running main"

python3 -u -c "import torch; print('GPU visible:', torch.cuda.is_available(), flush=True)"
python3 -u main_slurm.py
