#!/bin/bash
#SBATCH --output=/home-mscluster/erex/research_project/baseline/result.txt
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --partition=biggpu

pwd
date

scp ~/research_project/file.tar.gz /scratch/erex/Covidx-CT
ls
date
# rsync -a /tmp/baseline/ ~/research_project/baseline
