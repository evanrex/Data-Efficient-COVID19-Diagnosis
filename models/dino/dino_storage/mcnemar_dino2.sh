#!/bin/bash
#SBATCH --job-name=mcnmr2_dino
#SBATCH --output=/home-mscluster/erex/research_project/dino/mcnemar_result2.txt
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --partition=batch

python3 -u mcnemar_dino2.py
