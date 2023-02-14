#!/bin/bash
#SBATCH --job-name=mcnmr3_dino
#SBATCH --output=/home-mscluster/erex/research_project/dino/mcnemar_result3.txt
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --partition=batch

python3 -u mcnemar_dino3.py
