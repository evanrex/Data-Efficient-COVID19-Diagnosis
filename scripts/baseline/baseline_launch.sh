#!/bin/bash
#SBATCH --output=/home-mscluster/erex/research_project/baseline/result.txt
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --partition=stampede

pwd
date
python3 baseline_torch.py
date
