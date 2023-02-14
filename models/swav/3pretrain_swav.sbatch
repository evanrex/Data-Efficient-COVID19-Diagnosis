#!/bin/bash
#SBATCH --job-name=swav
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --partition=bigbatch
#SBATCH --output=/home-mscluster/erex/research_project/models/swav/experiments/result-%A-%a.out
#SBATCH --array=3-3 # job array index

echo "running job: "${SLURM_ARRAY_TASK_ID}
mkdir -p /home-mscluster/erex/research_project/models/swav/experiments/saving_dir${SLURM_ARRAY_TASK_ID}


python -m torch.distributed.launch --nproc_per_node=1 main_swav.py \
--data_path /home-mscluster/erex/research_project/datasets/NLST_dataset \
--dump_path /home-mscluster/erex/research_project/models/swav/experiments/saving_dir${SLURM_ARRAY_TASK_ID} \
--epochs 10 \
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
