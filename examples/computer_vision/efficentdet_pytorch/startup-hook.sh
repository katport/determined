cp -pv /run/determined/workdir/patch/__init__.py /run/determined/pythonuserbase/lib/python3.6/site-packages/determined/pytorch/__init__.py
cp -pv /run/determined/workdir/patch/_experimental.py /run/determined/pythonuserbase/lib/python3.6/site-packages/determined/pytorch/_experimental.py

cp -pv /run/determined/workdir/patch/_pytorch_trial.py /run/determined/pythonuserbase/lib/python3.6/site-packages/determined/pytorch/_pytorch_trial.py
cp -pv /run/determined/workdir/patch/_trial_controller.py /run/determined/pythonuserbase/lib/python3.6/site-packages/determined/_trial_controller.py

cp -pv /run/determined/workdir/patch/pytorch_context.py /run/determined/pythonuserbase/lib/python3.6/site-packages/determined/pytorch/_pytorch_context.py
cp -pv /run/determined/workdir/patch/reducer.py /run/determined/pythonuserbase/lib/python3.6/site-packages/determined/pytorch/_reducer.py


# echo HOROVOD_CYCLE_TIME
# export HOROVOD_CYCLE_TIME=2
pip install timm
git clone https://github.com/rwightman/efficientdet-pytorch.git
cd efficientdet-pytorch
git checkout 611532db49fdd691f48f913bc433391a12014bd8
python setup.py install
cd ..

