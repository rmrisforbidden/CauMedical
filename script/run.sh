ROOT_DIR='./output/causalMRI'
DATASET='brainweb' #'cifar10' #
DATA_DIR='/data/cifar10'

SEED=1234
GPU=3
OPTIM='adam'
LR=1e-3
EPOCH=100
WEIGHT_DECAY=4e-4
BATCH=128




CUDA_VISIBLE_DEVICES=$GPU python script/main.py --dataset $DATASET --data_dir $DATA_DIR --default_root_dir $ROOT_DIR \
--max_epochs $EPOCH --gpus 1 --seed $SEED --batch_size $BATCH --batch_size_ $BATCH \

