#!/bin/bash

python3 ../train.py \
    --data_dir '/content/drive/My Drive/atdt' \
    --model_dir '/content/drive/My Drive/Depth/experiments/depth_resnet50' \
    --checkpoint_dir '/content/drive/My Drive/Depth/experiments/depth_resnet50/ckpt' \
    --tensorboard_dir '/content/drive/My Drive/Depth/experiments/depth_resnet50/tensorboard' \
    --txt_train '/content/drive/My Drive/atdt/input_list_train_mixed_carla_cityscapes.txt' \
    --txt_val '/content/drive/My Drive/atdt/input_list_val_carla.txt'