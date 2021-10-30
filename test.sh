#SBATCH --partition=gpu
#SBATCH --time=0-12:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=16GB

cd /home/nameer/EECS542/selfsupervised-tracking/
module load python/3.8.7
source ../crw/bin/activate

export TEST_PY="code/test.py"
export CONVERT_PY="code/eval/convert_davis.py"
export DAVIS_FILE_PATH="../data/DAVIS/davis_vallist.txt"
export DAVIS_PATH="../data/davis/"
export DAVIS_EVAL_PATH="../data/davis2017-evaluation/evaluation_method.py"


#Paths you might actually want to change
export RESULTS_PATH="../evaluations/results1"
export PRETRAINED_PATH="../modelsaves/pretrained.pth"

python $TEST_PY --filelist $DAVIS_FILE_PATH \
    --model-type scratch --resume $PRETRAINED_PATH --save-path $RESULTS_PATH \
    --topk 10 --videoLen 20 --radius 12  --temperature 0.05 --cropSize 480


python $CONVERT_PY \
     --in_folder SAVE_PATH --out_folder $RESULTS_PATH --dataset $DAVIS_PATH

python $DAVIS_EVAL_PATH \
--task semi-supervised   --results_path $RESULTS_PATH --set val \
--davis_path $DAVIS_PATH