cp -pv /run/determined/workdir/local/__init__.py /run/determined/pythonuserbase/lib/python3.6/site-packages/determined/pytorch/__init__.py
cp -pv /run/determined/workdir/local/_data.py /run/determined/pythonuserbase/lib/python3.6/site-packages/determined/pytorch/_data.py

cp -pv /run/determined/workdir/local/_pytorch_context.py /run/determined/pythonuserbase/lib/python3.6/site-packages/determined/pytorch/_pytorch_context.py
cp -pv /run/determined/workdir/local/_pytorch_trial.py /run/determined/pythonuserbase/lib/python3.6/site-packages/determined/pytorch/_pytorch_trial.py

cp -pv /run/determined/workdir/local/_train_context.py /run/determined/pythonuserbase/lib/python3.6/site-packages/determined/_train_context.py
cp -pv /run/determined/workdir/local/samplers.py /run/determined/pythonuserbase/lib/python3.6/site-packages/determined/pytorch/samplers.py

export DETECTRON2_DATASETS=/mnt/dtrain-fsx/detectron2


# wget -O - https://fsx-lustre-client-repo-public-keys.s3.amazonaws.com/fsx-ubuntu-public-key.asc | apt-key add -
# bash -c 'echo "deb https://fsx-lustre-client-repo.s3.amazonaws.com/ubuntu xenial main" > /etc/apt/sources.list.d/fsxlustreclientrepo.list && apt-get update'
# apt install -y lustre-client-modules-$(uname -r)
# # Mount Lustre to /home/ubuntu/dtrain-fsx
# mkdir /home/ubuntu/dtrain-fsx
# mount -t lustre -o noatime,flock fs-0070d44fe131979c5.fsx.us-east-1.amazonaws.com@tcp:/moye7bmv /home/ubuntu/dtrain-fsx


# Uncomment if running without a container:
# pip install cython
# pip install pycocotools
# # # python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
# python -m pip install 'git+https://github.com/katport/detectron2_fork.git@e26ceda0a8c8ce68337926a767dbf5a86e215335'

# git clone https://github.com/facebookresearch/detectron2.git

# cd detectron2
# cd datasets
# mkdir coco
# mkdir coco/annotations
# cd coco/annotations

# # wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# cd ../../../../

# ./prepare_for_tests.sh
