#!/bin/bash

EVAL_DIR=./logs/
# CHECKPOINT_PATH=./checkpoints_VGG/VGG_VOC0712_SSD_300x300_iter_120000.ckpt
CHECKPOINT_PATH=./logs/model.ckpt-10518
python eval_ssd_network.py \
    --eval_dir=${EVAL_DIR} \
    --dataset_dir=./tfrecords \
    --dataset_name=pascalvoc_2007 \
    --dataset_split_name=test \
    --model_name=ssd_300_vgg \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --batch_size=1
