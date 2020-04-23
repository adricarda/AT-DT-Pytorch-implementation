#!/bin/bash

python3 ../evaluate.py \
    --data_dir '/content/drive/My Drive/atdt' \
    --model_dir '/content/drive/My Drive/Depth/experiments/depth_resnet50' \
    --txt_val '/content/drive/My Drive/atdt/input_list_val_carla.txt' \
    --checkpoint_dir '/content/drive/My Drive/Depth/experiments/depth_resnet50'