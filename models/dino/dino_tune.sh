#!/bin/bash
#SBATCH --job-name=dino_tune
#SBATCH --output=/home-mscluster/erex/research_project/dino/tune_result.txt
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --partition=batch

python3 -u dino_tune.py
