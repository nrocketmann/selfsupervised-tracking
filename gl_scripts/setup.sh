git clone https://github.com/davisvideochallenge/davis2017-evaluation.git
cd selfsupervised-tracking
git checkout gnn
git fetch
git reset --hard origin/gnn
cd ..
module load python/3.8.7
virtualenv -p python3.8.7 crw_env
source crw_env/bin/activate
pip install -r selfsupervised-tracking/requirements.txt
pip install gdown
