ROOT_DIR='./output/Split'
DATA_DIR='/data/cifar10'

SEED=1234
GPU=2
# WEIGHT_DECAY=4e-4
BATCH=1000
DECODER='..' #'simple' #bloch



########################## DATA_TYPE List ####################################
# Trainset
## seqTr, seqAll, phantomTr, phantomAll

# Testset
## seqTe, phantomTe, phantomAll, real


###############################################################


########################## Test for Dictionary based ####################################
# TR_DATA_TYPE='seqTr'
# TE_DATA_TYPE='seqTe'
# CUDA_VISIBLE_DEVICES=$GPU python lightning_bolts/script/main_test_img.py --default_root_dir $ROOT_DIR \
# --train_data_type $TR_DATA_TYPE --test_data_type $TE_DATA_TYPE --batch_size $BATCH --is_dictionary_based 1
###############################################################

# TR_DATA_TYPE='seqTr'
# TE_DATA_TYPE='phantomTe'
# CUDA_VISIBLE_DEVICES=$GPU python lightning_bolts/script/main_test_img.py --default_root_dir $ROOT_DIR \
# --train_data_type $TR_DATA_TYPE --test_data_type $TE_DATA_TYPE --batch_size $BATCH --is_dictionary_based 1

TR_DATA_TYPE='seqAll_RF_4SplineNoise11Flex_num200_num250' #'seqAll_RF_4SplineNoise11Flex_num200_1Spline5Flex_num200' #'seqAll_RF_1Spline5Flex_num200_num250'
CUDA_VISIBLE_DEVICES=$GPU python lightning_bolts/script/main_test_img.py --default_root_dir $ROOT_DIR \
--train_data_type $TR_DATA_TYPE --batch_size $BATCH --is_dictionary_based 1

# TR_DATA_TYPE='seqAll_RF_1Spline5Flex_num200_num250'
# CUDA_VISIBLE_DEVICES=$GPU python lightning_bolts/script/main_test_img.py --default_root_dir $ROOT_DIR \
# --train_data_type $TR_DATA_TYPE --batch_size $BATCH --is_dictionary_based 1

# TR_DATA_TYPE='phantomTr'
# TE_DATA_TYPE='phantomTe'
# CUDA_VISIBLE_DEVICES=$GPU python lightning_bolts/script/main_test_img.py --default_root_dir $ROOT_DIR \
# --train_data_type $TR_DATA_TYPE --test_data_type $TE_DATA_TYPE --batch_size $BATCH --is_dictionary_based 1

# TR_DATA_TYPE='phantomAll'
# TE_DATA_TYPE='real'
# CUDA_VISIBLE_DEVICES=$GPU python lightning_bolts/script/main_test_img.py --default_root_dir $ROOT_DIR \
# --train_data_type $TR_DATA_TYPE --test_data_type $TE_DATA_TYPE --batch_size $BATCH --is_dictionary_based 1
