#!/bin/bash
#SBATCH --job-name=byol
#SBATCH --output=/home-mscluster/erex/research_project/byol/result.txt
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --partition=batch

date
python -u main_byol.py
