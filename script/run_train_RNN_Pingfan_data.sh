ROOT_DIR='./output/SeqBlochDecoder'
DATA_DIR='/data/cifar10'

SEED=1234
GPU=3
OPTIM='adam'
LR=1e-3
# WEIGHT_DECAY=4e-4
BATCH=200
DECODER='bloch' #'simple' #

for IS_RF in 1
do
for MODEL_TYPE in RNN
do
for EPOCH in 200 #3000
do
CUDA_VISIBLE_DEVICES=$GPU python lightning_bolts/script/approx_main.py --default_root_dir $ROOT_DIR \
--gpus 1 --seed $SEED --batch_size $BATCH --num_workers 4 \
--max_epochs $EPOCH --approx_model_type $MODEL_TYPE --is_input_RF $IS_RF
done
done
done