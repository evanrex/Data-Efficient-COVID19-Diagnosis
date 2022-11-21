#!/bin/bash
#SBATCH --job--name=data_hand
#SBATCH --output=/home-mscluster/erex/research_project/result.txt
#SBATCH --ntasks=1
#SBATCH --time=6:00:00
#SBATCH --partition=stampede

date
rsync -a ~/research_project/covidx-ct/ /tmp/covidx-ct/
date
mkdir Covidx-CT/train
mkdir Covidx-CT/train/positive
mkdir Covidx-CT/train/negative
mkdir Covidx-CT/validation
mkdir Covidx-CT/validation/positive
mkdir Covidx-CT/validation/negative
mkdir Covidx-CT/test
mkdir Covidx-CT/test/positive
mkdir Covidx-CT/test/negative
ls Covidx-CT

python3 covid_datahandle.py
date
rsync -a  /tmp/Covidx-CT/ ~/research_project/Covidx-CT/
date
