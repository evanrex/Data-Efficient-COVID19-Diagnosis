#!/bin/bash
#SBATCH --job-name=mcnmr_dino
#SBATCH --output=/home-mscluster/erex/research_project/dino/mcnemar_result.txt
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --partition=batch

python3 -u mcnemar_dino.py