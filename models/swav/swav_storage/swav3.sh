#!/bin/bash
#SBATCH --job-name=swav
#SBATCH --output=/home-mscluster/erex/research_project/swav/result3.txt
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --partition=batch

python -m torch.distributed.launch --nproc_per_node=1 main_swav.py \
--data_path /home-mscluster/erex/research_project/NLST_dataset \
--dump_path /home-mscluster/erex/research_project/swav/saving_dir3 \
--epochs 100 \
--base_lr 0.6 \
--final_lr 0.0006 \
--warmup_epochs 0 \
--batch_size 32 \
--size_crops 224 96 \
--nmb_crops 2 6 \
--min_scale_crops 0.14 0.05 \
--max_scale_crops 1. 0.14 \
--use_fp16 false \
--freeze_prototypes_niters 5005 \
--queue_length 3840 \
--epoch_queue_starts 15
