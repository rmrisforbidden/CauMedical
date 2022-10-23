ROOT_DIR='./output/Split'
DATA_DIR='/data/cifar10'

SEED=1234
GPU=0
OPTIM='adam'
LR=1e-3
# WEIGHT_DECAY=4e-4
BATCH=1000

for IS_RF in 1
do
for MODEL_TYPE in RNN
do
for EPOCH in 50
do
for TR_DATA_TYPE in 'seqTr'
do
for TE_DATA_TYPE in 'seqTe'
do
CUDA_VISIBLE_DEVICES=$GPU python lightning_bolts/script/approx_main.py --default_root_dir $ROOT_DIR \
--gpus 1 --seed $SEED --batch_size $BATCH --num_workers 4 \
--max_epochs $EPOCH --approx_model_type $MODEL_TYPE --is_input_RF $IS_RF \
--train_data_type $TR_DATA_TYPE --test_data_type $TE_DATA_TYPE
done
done
done
done
done

