#!/bin/bash
#SBATCH --job-name=byol2_tune
#SBATCH --output=/home-mscluster/erex/research_project/byol/tune_result2.txt
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --partition=batch

python3 -u byol_tune2.py
