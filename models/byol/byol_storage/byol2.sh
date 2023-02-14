#!/bin/bash
#SBATCH --job-name=byol
#SBATCH --output=/home-mscluster/erex/research_project/byol/result2.txt
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --partition=batch

date
python hello_world.py
python -u main2.py
