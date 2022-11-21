#!/bin/bash
#SBATCH --job-name=dino
#SBATCH --output=/home-mscluster/erex/research_project/dino/result.txt
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --partition=batch

python -m torch.distributed.launch --nproc_per_node=1 main_dino.py --epochs 1 --arch resnet50 --optimizer sgd --lr 0.03 --weight_decay 1e-4 --weight_decay_end 1e-4 --global_crops_scale 0.14 1 --local_crops_scale 0.05 0.14 --data_path /home-mscluster/erex/research_project/NLST_dataset --output_dir /home-mscluster/erex/research_project/dino/saving_dir