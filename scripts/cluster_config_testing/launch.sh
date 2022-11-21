#!/bin/bash
#SBATCH --job--name=test
#SBATCH --output=/home-mscluster/erex/test/result.txt
#SBATCH --ntasks=1
#SBATCH --time=10:00
#SBATCH --partition=stampede
python3 hello_world.py
