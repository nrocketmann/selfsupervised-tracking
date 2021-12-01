#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --time=0-01:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=32
#SBATCH --mem-per-gpu=16GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=duttar@umich.edu
#SBATCH --job-name=entropy-scratch
#SBATCH --account=eecs542f21_class
cd ~
module load python/3.8.7
source crw_env/bin/activate
export TRAIN_PY="/home/duttar/selfsupervised-tracking/code/train.py"
export DATA_PATH="/home/duttar/vos_clips/"
export CACHE_PATH="/home/duttar/vos_all_clips"
export FINETUNE_PATH="/home/duttar/selfsupervised-tracking/pretrained_enc_fc.pth"
export OUTPUT_PATH="/home/duttar/entropy_output/"
mkdir $OUTPUT_PATH
python -u -W ignore $TRAIN_PY --data-path $DATA_PATH \
        --frame-aug grid --dropout 0.1 --clip-len 4 --temp 0.05 \
        --model-type scratch  --workers 32 --batch-size 12 \
        --load-cache $CACHE_PATH --data-parallel --lr 0.0001 \
        --output-dir $OUTPUT_PATH --epochs 200 --lr-milestones 250 \
        --finetune $FINETUNE_PATH