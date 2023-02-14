#!/bin/bash
#SBATCH --job-name=swav_tune
#SBATCH --output=/home-mscluster/erex/research_project/swav/tune_result.txt
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --partition=batch

python3 -u swav_tune.py
