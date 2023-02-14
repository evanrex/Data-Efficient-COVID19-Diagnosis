#!/bin/bash
#SBATCH --job-name=ROC3_dino
#SBATCH --output=/home-mscluster/erex/research_project/dino/ROC_result3.txt
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --partition=stampede

python3 -u ROC_dino3.py
