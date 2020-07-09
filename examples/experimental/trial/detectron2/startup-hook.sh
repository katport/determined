# pip install cython
# pip install pycocotools
# python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

git clone https://github.com/facebookresearch/detectron2.git

cd detectron2
cd datasets
mkdir coco
mkdir coco/annotations
cd coco/annotations

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

cd ../../../../

export DETECTRON2_DATASETS=/mnt/dtrain-fsx/detectron2
# ./prepare_for_tests.sh
