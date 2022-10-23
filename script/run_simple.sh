ROOT_DIR='./output/SeqBlochDecoder'
DATA_DIR='/data/cifar10'

SEED=1234
GPU=2
OPTIM='adam'
LR=1e-3
# WEIGHT_DECAY=4e-4
BATCH=50
DECODER='bloch' #'simple' #



# for DATASET in seq
# do
# for DECODER in none
# do
# for LAMB in 0
# do
# EPOCH=300
# CUDA_VISIBLE_DEVICES=$GPU python lightning_bolts/script/simple_main.py --test_dataset $DATASET --default_root_dir $ROOT_DIR \
# --gpus 1 --seed $SEED --batch_size $BATCH --num_workers 4 \
# --max_epochs $EPOCH --decoder_type $DECODER --rec_lambda $LAMB
# done
# done
# done

for DATASET in seq
do
for DECODER in bloch
do
for LAMB in 1000 1
do
EPOCH=100
MIXED=80
CUDA_VISIBLE_DEVICES=$GPU python lightning_bolts/script/simple_main.py --test_dataset $DATASET --default_root_dir $ROOT_DIR \
--gpus 1 --seed $SEED --batch_size $BATCH --num_workers 4 \
--max_epochs $EPOCH --decoder_type $DECODER --rec_lambda $LAMB --mixed_training $MIXED
done
done
done

