#!/bin/bash -x
#SBATCH --job-name=translate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --time=22:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=<user>@princeton.edu
#SBATCH --output=/scratch/gpfs/your-net-id/some-project-dir/translate.out

cd /scratch/gpfs/your-net-id/some-project-dir

uv run python -u translate_documents.py ./input --output-dir ./output --backend huggingface --batch-size 25 --model google/translategemma-27b-it --cache-dir ./.hf