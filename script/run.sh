ROOT_DIR='./output/Split'
DATA_DIR='/data/cifar10'

SEED=1234
GPU=1
# WEIGHT_DECAY=4e-4
BATCH=1000
DECODER='..' #'simple' #bloch

############################# Original ##################################
# for EXP_ID in SPLIT-80 SPLIT-90 SPLIT-101 SPLIT-88 SPLIT-71
# do
# for DATA_TYPE in img_none #seq_seq #img_img seq_img
# do
# CUDA_VISIBLE_DEVICES=$GPU python lightning_bolts/script/main_test_img.py --default_root_dir $ROOT_DIR \
# --data_type $DATA_TYPE --batch_size $BATCH --exp_id $EXP_ID
# done
# done
# done
# done
# done
###############################################################





########################### This is for None ####################################
# DATA_TYPE='img_none' #'real_img'
# EXP_ID="SPLIT-142"
# CUDA_VISIBLE_DEVICES=$GPU python lightning_bolts/script/main_test_img.py --default_root_dir $ROOT_DIR \
# --data_type $DATA_TYPE --batch_size $BATCH --exp_id $EXP_ID
###############################################################

########################## WHen lots of IDs ####################################


for TE_DATA_TYPE in 'real' #'real' #phantomAll
do
for EXP_ID in  "SPLIT-225" #217 
do

CUDA_VISIBLE_DEVICES=$GPU python lightning_bolts/script/main_test_img.py --default_root_dir $ROOT_DIR \
--test_data_type $TE_DATA_TYPE --batch_size $BATCH --exp_id $EXP_ID

done
done
###############################################################
