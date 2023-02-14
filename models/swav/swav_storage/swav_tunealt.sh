#!/bin/bash
#SBATCH --job-name=swavalt_tune
#SBATCH --output=/home-mscluster/erex/research_project/swav/tune_resultalt.txt
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --partition=batch

python3 -u swav_tunealt.py
