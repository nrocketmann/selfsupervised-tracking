#SBATCH --partition=gpu
#SBATCH --time=0-20:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=32
#SBATCH --mem-per-gpu=16GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nameer@umich.edu
#SBATCH --job-name=train-nndustbin
#SBATCH --account=eecs542f21_class
module load python/3.8.7
source crw_env/bin/activate
export TRAIN_PY="selfsupervised-tracking/code/train.py"
export DATA_PATH="clips/"
export CACHE_PATH="vos_all_clips"
export OUTPUT_PATH="nndustbin_scratch1/"
mkdir $OUTPUT_PATH
python -u -W ignore $TRAIN_PY --data-path $DATA_PATH \
        --frame-aug grid --dropout 0.1 --clip-len 4 --temp 0.05 \
        --model-type scratch  --workers 32 --batch-size 12 \
        --load-cache $CACHE_PATH --data-parallel --lr 0.0001 \
        --output-dir $OUTPUT_PATH --epochs 200 --lr-milestones 250 \
        --resume selfsupervised-tracking/pretrained_enc_fc.pth