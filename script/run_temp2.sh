ROOT_DIR='./output/Split'
DATA_DIR='/data/cifar10'

SEED=1234
GPU=1
OPTIM='adam'
LR=1e-3
# WEIGHT_DECAY=4e-4
BATCH=1000
DECODER='..' #'simple' #bloch
IS_EMB_LOSS=1


for TR_DATA_TYPE in 'seqAll_RF_4SplineNoise11Flex_num200_num250' #'seqAll_RF_4SplineNoise11Flex_num200_1Spline5Flex_num200' 
do
for IS_EMB_LOSS in 1
do
for EPOCH in 26
do
for DECODER in causal_RNN
do
for LAMB in 1 #10 0.1 #100 500 1000 1 10 5000 10000
do
for ENCODER in FC
do
CUDA_VISIBLE_DEVICES=$GPU python lightning_bolts/script/main_train_enc_with_our_RNN.py --default_root_dir $ROOT_DIR \
--gpus 1 --seed $SEED --batch_size $BATCH --num_workers 4 \
--max_epochs $EPOCH --decoder_type $DECODER --rec_lambda $LAMB \
--is_emb_loss $IS_EMB_LOSS \
--train_data_type $TR_DATA_TYPE --encoder_type $ENCODER
done
done
done
done
done
done

# for TR_DATA_TYPE in 'seqAll_RF_4SplineNoise11Flex_num200_1Spline5Flex_num200' 'seqAll_RF_1Spline5Flex_num200_num250' 
# do
# for IS_EMB_LOSS in 1
# do
# for EPOCH in 26
# do
# for DECODER in simple_FC simple_RNN #none #causal_RNN
# do
# for LAMB in 10 #10 0.1 #100 500 1000 1 10 5000 10000
# do
# for ENCODER in FC
# do
# CUDA_VISIBLE_DEVICES=$GPU python lightning_bolts/script/main_train_enc_with_our_RNN.py --default_root_dir $ROOT_DIR \
# --gpus 1 --seed $SEED --batch_size $BATCH --num_workers 4 \
# --max_epochs $EPOCH --decoder_type $DECODER --rec_lambda $LAMB \
# --is_emb_loss $IS_EMB_LOSS \
# --train_data_type $TR_DATA_TYPE --encoder_type $ENCODER
# done
# done
# done
# done
# done
# done

# for TR_DATA_TYPE in 'seqAll' 
# do
# for IS_EMB_LOSS in 1
# do
# for EPOCH in 26
# do
# for DECODER in simple_FC simple_RNN #none #causal_RNN
# do
# for LAMB in 10 #10 0.1 #100 500 1000 1 10 5000 10000
# do
# for ENCODER in FC RNN
# do
# CUDA_VISIBLE_DEVICES=$GPU python lightning_bolts/script/main_train_enc_with_our_RNN.py --default_root_dir $ROOT_DIR \
# --gpus 1 --seed $SEED --batch_size $BATCH --num_workers 4 \
# --max_epochs $EPOCH --decoder_type $DECODER --rec_lambda $LAMB \
# --is_emb_loss $IS_EMB_LOSS \
# --train_data_type $TR_DATA_TYPE --encoder_type $ENCODER
# done
# done
# done
# done
# done
# done

