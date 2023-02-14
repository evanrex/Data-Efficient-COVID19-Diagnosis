#!/bin/bash
#SBATCH --job-name=byol_tune
#SBATCH --output=/home-mscluster/erex/research_project/byol/tune_result.txt
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --partition=batch

python3 -u byol_tune.py
