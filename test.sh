PRETRAINED_PATH = "../modelsaves/pretrained.pth"
TEST_PY = "code/test.py"
CONVERT_PY = "code/eval/convert_davis.py"
DAVIS_FILE_PATH = "../data/DAVIS/davis_vallist.txt"
SAVE_PATH = "../evaluations/results1"
CONVERT_PATH = "../evaluations/results1_converted"
DAVIS_PATH = "../davis/"

python $TEST_PY --filelist $DAVIS_FILE_PATH \
    --model-type scratch --resume $PRETRAINED_PATH --save-path $SAVE_PATH \
    --topk 10 --videoLen 20 --radius 12  --temperature 0.05 --cropSize 480


python $CONVERT_PY \
     --in_folder SAVE_PATH --out_folder $CONVERT_PATH --dataset $DAVIS_PATH

python davis2017-evaluation/evaluation_method.py \
--task semi-supervised   --results_path davis_final6/ --set val \
--davis_path davis/