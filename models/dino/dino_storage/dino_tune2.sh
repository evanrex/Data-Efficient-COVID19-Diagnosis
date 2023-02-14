#!/bin/bash
#SBATCH --job-name=dino2
#SBATCH --output=/home-mscluster/erex/research_project/dino/tune_result2.txt
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --partition=batch

python3 -u dino_tune2.py
