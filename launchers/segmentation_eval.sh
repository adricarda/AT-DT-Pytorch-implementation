#!/bin/bash

python3 ../evaluate.py \
    --data_dir '/content/drive/My Drive/atdt' \
    --model_dir '/content/drive/My Drive/Depth/experiments/segmentation_resnet50' \
    --checkpoint_dir '/content/drive/My Drive/Depth/experiments/segmentation_resnet50/ckpt' \
    --txt_val '/content/drive/My Drive/atdt/input_list_val_carla.txt'