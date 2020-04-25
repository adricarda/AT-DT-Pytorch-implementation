#!/bin/bash

python3 ../lr_finder.py \
    --data_dir '/content/drive/My Drive/atdt' \
    --model_dir '/content/drive/My Drive/Depth/experiments/segmentation_resnet50' \
    --checkpoint_dir '/content/drive/My Drive/Depth/experiments/segmentation_resnet50/ckpt' \
    --txt_train '/content/drive/My Drive/atdt/input_list_train_carla.txt'