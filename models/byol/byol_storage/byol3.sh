#!/bin/bash
#SBATCH --job-name=byol3
#SBATCH --output=/home-mscluster/erex/research_project/byol/result3.txt
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --partition=batch

date
python hello_world.py
python -u main3.py
