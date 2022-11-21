#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=/home-mscluster/erex/test/result.txt
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --partition=batch

python3 ~/test/hello_world.py
