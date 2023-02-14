#!/bin/bash
#SBATCH --job-name=dino%A
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --partition=biggpu
#SBATCH --output=/home-mscluster/erex/research_project/models/dino/experiments/result-%A-%a.out
#SBATCH --array=3-5 # job array index


echo "running job: "${SLURM_ARRAY_TASK_ID}
mkdir -p /home-mscluster/erex/research_project/models/dino/experiments/saving_dir${SLURM_ARRAY_TASK_ID}

python3 -m torch.distributed.run --nproc_per_node=2 main_dino.py --epochs 10 --arch resnet50 --optimizer sgd --lr 0.03 --weight_decay 1e-4 --weight_decay_end 1e-4 --global_crops_scale 0.14 1 --local_crops_scale 0.05 0.14 --data_path /home-mscluster/erex/research_project/datasets/NLST_dataset --output_dir /home-mscluster/erex/research_project/models/dino/experiments/saving_dir${SLURM_ARRAY_TASK_ID}
