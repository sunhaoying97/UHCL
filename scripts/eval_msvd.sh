# Setup
DATATYPE=msvd
N_GPU=1 ##2
N_THREAD=16

DATA_PATH=../dataset/MSVD
CKPT_ROOT=../Result
INIT_MODEL_PATH=/media/alocus/4A984F62984F4C1F/Users/Alocus/Desktop/视频字幕/UHCL/ckpts/msvd_lr7e-6/VIT-B-32.pt111.12


#../weight/univl.pretrained.bin
# Path to the features you extracted from CLIP4Clip
FEATURES_PATH=../extracted_feats/msvd/MSVD_CLIP4Clip_features.pickle
# Tuning Params
LEARNING_RATE=7e-6
#--batch_size=128


python -m torch.distributed.launch --nproc_per_node=${N_GPU} \
../train.py --do_eval --num_thread_reader=${N_THREAD} \
--epochs=50 --batch_size=1 --n_display=50 --gradient_accumulation_steps 2 \
--data_path ${DATA_PATH} --features_path ${FEATURES_PATH} \
--output_dir ${CKPT_ROOT}/${DATATYPE}_lr${lr} \
--bert_model bert-base-uncased --do_lower_case \
--lr ${LEARNING_RATE} --max_words 48 --max_frames 20 --batch_size_val 1 \
--visual_num_hidden_layers 2 --decoder_num_hidden_layers 2 \
--datatype ${DATATYPE} --init_model ${INIT_MODEL_PATH} \
--d_model 512 --video_dim 512
