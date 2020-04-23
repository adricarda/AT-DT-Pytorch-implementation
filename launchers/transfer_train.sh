#!/bin/bash

python3 ../train_transfer.py \
    --data_dir '/content/drive/My Drive/atdt' \
    --model_dir_source '/content/drive/My Drive/Depth/experiments/depth_resnet50' \
    --checkpoint_dir_source '/content/drive/My Drive/Depth/experiments/depth_resnet50' \
    --model_dir_target '/content/drive/My Drive/Depth/experiments/segmentation_resnet50' \
    --checkpoint_dir_target '/content/drive/My Drive/Depth/experiments/segmentation_resnet50' \
    --model_dir_transfer '/content/drive/My Drive/Depth/experiments/transfer_baseline' \
    --checkpoint_dir_transfer '/content/drive/My Drive/Depth/experiments/transfer_baseline' \
    --tensorboard_dir '/content/drive/My Drive/Depth/experiments/transfer_baseline/tensorboard' \
    --txt_train '/content/drive/My Drive/atdt/input_list_train_carla.txt' \
    --txt_val '/content/drive/My Drive/atdt/input_list_val_carla.txt'