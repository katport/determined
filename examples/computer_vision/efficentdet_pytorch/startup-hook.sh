# echo HOROVOD_CYCLE_TIME
# export HOROVOD_CYCLE_TIME=2
pip install timm
git clone https://github.com/rwightman/efficientdet-pytorch.git
cd efficientdet-pytorch
git checkout 611532db49fdd691f48f913bc433391a12014bd8
python setup.py install
cd ..

