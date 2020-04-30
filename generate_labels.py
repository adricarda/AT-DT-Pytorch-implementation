"""Train the model"""

import argparse
import copy
import logging
import os
import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random
from evaluate import evaluate
import dataloader.dataloader as dataloader
import utils.utils as utils
from model.losses import get_loss_fn
from model.metrics import get_metrics
from model.net import get_network, get_transfer, get_adaptive_network
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/content/drive/My Drive/atdt',
                    help="Directory containing the dataset")

parser.add_argument('--model_dir_source', default='experiments/depth_resnet50',
                    help="Directory containing params.json")
parser.add_argument('--model_dir_target', default='experiments/segmentation_resnet50',
                    help="Directory containing params.json")
parser.add_argument('--model_dir_transfer', default='experiments/transfer_baseline',
                    help="Directory containing params.json")

parser.add_argument('--checkpoint_dir_source', default="experiments/depth_resnet50/ckpt",
                    help="Directory containing source model weights")
parser.add_argument('--checkpoint_dir_target', default="experiments/segmentation_resnet50/ckpt",
                    help="Directory containing weights target model weights")
parser.add_argument('--checkpoint_dir_transfer', default="experiments/transfer_baseline/ckpt",
                    help="Directory containing weights target model weights")

parser.add_argument('--txt_val', default='/content/drive/My Drive/atdt/input_list_val_cityscapes.txt',
                    help="Txt file containing path to validation images")


def generate(model, dataset_dl, dir_path, file_list, params):

    # set model to evaluation mode
    model.eval()

    with torch.no_grad():
        for (xb, yb, path) in tqdm(dataset_dl):
            xb = xb.to(params.device)
            yb = yb.to(params.device)
            output = model(xb)['out']
            prediction = output.argmax(dim=1).squeeze()
            path = path[0].replace('.png', '_noisy_carla.png')
            # os.makedirs(os.path.dirname(path), exist_ok=True)
            cv2.imwrite(path, prediction.cpu().numpy())
            # print(path)
            
if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()

    json_path = os.path.join(args.model_dir_source, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params_source = utils.Params(json_path)

    json_path = os.path.join(args.model_dir_target, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params_target = utils.Params(json_path)

    json_path = os.path.join(args.model_dir_transfer, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params_transfer = utils.Params(json_path)

    # ckpt_filename = "checkpoint.tar"
    best_ckpt_filename = "model_best.tar"
    # writer = SummaryWriter(args.tensorboard_dir)

    # use GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    params_transfer.device = device

    # Set the random seed for reproducible experiments
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    params_transfer.encoding = params_transfer.encoding_target
    val_dl = dataloader.fetch_dataloader(
        args.data_dir, args.txt_val, 'val', params_transfer)

    # logging.info("- done.")

    # Define the model and optimizer
    model_source = get_network(params_source).to(params_transfer.device)
    model_target = get_network(params_target).to(params_transfer.device)
    transfer = get_transfer(params_transfer).to(params_transfer.device)

    #load source and target model before training and extract backbones
    ckpt_source_file_path = os.path.join(args.checkpoint_dir_source, best_ckpt_filename)
    if os.path.exists(ckpt_source_file_path):
        model_source = utils.load_checkpoint(model_source, ckpt_dir=args.checkpoint_dir_source, filename=best_ckpt_filename, is_best=True)[0]
        print("=> loaded source model checkpoint form {}".format(ckpt_source_file_path))
    else:
        print("=> Initializing source model from scratch")
    
    ckpt_target_file_path = os.path.join(args.checkpoint_dir_target, best_ckpt_filename)
    if os.path.exists(ckpt_target_file_path):
        model_target = utils.load_checkpoint(model_target, ckpt_dir=args.checkpoint_dir_target, filename=best_ckpt_filename, is_best=True)[0]
        print("=> loaded target model checkpoint form {}".format(ckpt_target_file_path))
    else:
        print("=> Initializing target model from scratch")
    
    ckpt_transfer_file_path = os.path.join(args.checkpoint_dir_transfer, best_ckpt_filename)
    if os.path.exists(ckpt_transfer_file_path):
        model_transfer = utils.load_checkpoint(transfer, ckpt_dir=args.checkpoint_dir_transfer, filename=best_ckpt_filename, is_best=True)[0]
        print("=> loaded transfer checkpoint form {}".format(ckpt_transfer_file_path))
    else:
        print("=> Initializing from scratch")
    
    metrics = OrderedDict({})
    for metric in params_transfer.metrics:
        metrics[metric] = get_metrics(metric, params_transfer)

    #construct graph adaptation model
    source_encoder = model_source.backbone
    target_decoder = model_target.classifier
    adpative_model = get_adaptive_network(source_encoder, model_transfer, target_decoder)

     # Evaluate
    generate(adpative_model, val_dl, '/content/drive/My Drive/cityscapes_noisy', args.txt_val, params_transfer)