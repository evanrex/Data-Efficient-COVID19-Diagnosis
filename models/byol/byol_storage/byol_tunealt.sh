#!/bin/bash
#SBATCH --job-name=byolalt_tune
#SBATCH --output=/home-mscluster/erex/research_project/byol/tune_resultalt.txt
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --partition=batch

python3 -u byol_tunealt.py
