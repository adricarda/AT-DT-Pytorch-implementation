"""Train the model"""

import argparse
import copy
import logging
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random
from evaluate import evaluate
import torch.nn.functional as F
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

parser.add_argument('--tensorboard_dir', default="experiments/transfer_baseline/tensorboard",
                    help="Directory for Tensorboard data")
parser.add_argument('--txt_train_carla', default='/content/drive/My Drive/atdt/input_list_train_carla.txt',
                    help="Txt file containing path to training images for carla")
parser.add_argument('--txt_train_cs', default='/content/drive/My Drive/atdt/input_list_train_cityscapes.txt',
                    help="Txt file containing path to training images for cityscapes")
parser.add_argument('--txt_val_source', default='/content/drive/My Drive/atdt/input_list_val_carla.txt',
                    help="Txt file containing path to validation images source dataset")
parser.add_argument('--txt_val_target', default='/content/drive/My Drive/atdt/input_list_val_cityscapes.txt',
                    help="Txt file containing path to validation images target dataset")


def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']


def inference(model, batch):
    model.eval()
    with torch.no_grad():
        y_pred = model(batch.to(device))['out']
    return y_pred


def train_step(model, batch_x, batch_y, opt, loss_fn):
    output = model(batch_x)['out']
    loss_b = loss_fn(output, batch_y)

    if opt is not None:
        opt.zero_grad()
        loss_b.backward()
        opt.step()

    return loss_b


def train_epoch(model_source, model_target, transfer, 
                train_dl_all, train_dl_depth_target, 
                opt1, opt2, opt3, loss_fn1, loss_fn2, params,
                lr_scheduler1, lr_scheduler2, lr_scheduler3):

    running_loss_depth_carla = utils.RunningAverage()

    iter_cs_depth = iter(train_dl_depth_target)
    iter_carla_depth_sem = iter(train_dl_all)

    source_encoder = model_source.backbone
    source_decoder = model_source.classifier
    target_decoder = model_target.classifier

    for (batch_images_carla, batch_segmentation_carla, batch_depth_carla) in tqdm(train_dl_all):
        input_shape = batch_images_carla.shape[-2:]

        try:
            batch_images_cs, batch_depth_cs = next(iter_cs_depth)
        except StopIteration:
            iter_cs_depth = iter(train_dl_depth_target)
            batch_images_cs, batch_depth_cs = next(iter_cs_depth)

        loss_cs_depth = train_step(model_source, batch_images_cs.to(params.device), batch_depth_cs.to(params.device), opt1, loss_fn1)

        batch_images_carla = batch_images_carla.to(params.device)
        batch_segmentation_carla = batch_segmentation_carla.to(params.device)
        batch_depth_carla = batch_depth_carla.to(params.device)

        depth_feature = source_encoder(batch_images_carla)['out']
        depth_feature_copy = depth_feature.detach()
        depth_prediction = source_decoder(depth_feature)

        depth_prediction = F.interpolate(depth_prediction, size=input_shape, mode='bilinear', align_corners=False)
        loss_depth_carla = loss_fn1(depth_prediction, batch_depth_carla)

        if opt1 is not None:
            opt1.zero_grad()
            loss_depth_carla.backward()
            opt1.step()

        if lr_scheduler1 is not None:
            lr_scheduler1.step()       

        loss_carla_segementation = train_step(model_target, batch_images_carla, batch_segmentation_carla, opt2, loss_fn2)
        if lr_scheduler2 is not None:
            lr_scheduler2.step()

        adapted_feature = transfer(depth_feature_copy)
        adapted_carla_prediction = target_decoder(adapted_feature)
        adapted_carla_prediction = F.interpolate(adapted_carla_prediction, size=input_shape, mode='bilinear', align_corners=False)
        loss_adapted_carla = loss_fn2(adapted_carla_prediction, batch_segmentation_carla)

        if opt3 is not None:
            opt3.zero_grad()
            loss_adapted_carla.backward()
            opt3.step()

        if lr_scheduler3 is not None:
            lr_scheduler3.step()

        running_loss_depth_carla.update(loss_depth_carla.item())

    return running_loss_depth_carla()


def train_and_evaluate(model_source, model_target, transfer, train_dl_all, train_dl_depth_target, val_dl_source_all, val_dl_target, 
                        opt1, opt2, opt3, loss_fn1, loss_fn2, metrics, params,
                        lr_scheduler1, lr_scheduler2, lr_scheduler3,
                        checkpoint_dir, ckpt_filename, log_dir, writer):

    ckpt_file_path = os.path.join(checkpoint_dir, ckpt_filename)
    best_value = -float('inf')
    early_stopping = utils.EarlyStopping(patience=10, verbose=True)
    start_epoch = 0

    batch_sample_carla, batch_gt_carla_sem, batch_gt_carla_depth = next(iter(val_dl_source_all))
    batch_sample_cs, batch_gt_cs = next(iter(val_dl_target))

    source_encoder = model_source.backbone
    target_decoder = model_target.classifier
    adpative_model = get_adaptive_network(source_encoder, transfer, target_decoder)

    for epoch in range(start_epoch, params.num_epochs):
        # Run one epoch
        current_lr = get_lr(opt1)
        logging.info('Epoch {}/{}, current lr={}'.format(epoch,
                                                         params.num_epochs-1, current_lr))
        writer.add_scalar('Learning_rate', current_lr, epoch)

        if epoch % 5 == 0 or epoch==params.num_epochs-1:
            predictions_sem = inference(model_target, batch_sample_carla)
            predictions_depth = inference(model_source, batch_sample_carla)

            plot = train_dl_all.dataset.get_predictions_plot(
                batch_sample_carla, predictions_sem.cpu(), batch_gt_carla_sem.cpu(), predictions_depth.cpu(), batch_gt_carla_depth.cpu())
            writer.add_image('Predictions_carla', plot, epoch, dataformats='HWC')

            predictions = inference(adpative_model, batch_sample_cs)
            plot = val_dl_target.dataset.dataset.get_predictions_plot(
                batch_sample_cs, predictions.cpu(), batch_gt_cs)
            writer.add_image('Predictions_target', plot, epoch, dataformats='HWC')

        transfer.train()
        train_loss = train_epoch(
                        model_source, model_target, transfer, 
                        train_dl_all, train_dl_depth_target, 
                        opt1, opt2, opt3, loss_fn1, loss_fn2, params,
                        lr_scheduler1, lr_scheduler2, lr_scheduler3)

        logging.info("-"*20)


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

    ckpt_filename = "checkpoint.tar"
    best_ckpt_filename = "model_best.tar"
    writer = SummaryWriter(args.tensorboard_dir)

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

    # Set the logger
    log_dir = os.path.join(args.model_dir_transfer, "logs")
    if not os.path.exists(log_dir):
        print("Making log directory {}".format(log_dir))
        os.mkdir(log_dir)
    utils.set_logger(os.path.join(log_dir, "train.log"))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
    params_transfer.encoding = params_transfer.encoding_source
    train_dl_all = dataloader.fetch_dataloader(
        args.data_dir, args.txt_train_carla, 'train', params_transfer, sem_depth=True)

    train_dl_depth_target = dataloader.fetch_dataloader(
        args.data_dir, args.txt_train_cs, 'train', params_source)

    val_dl_source_all = dataloader.fetch_dataloader(
        args.data_dir, args.txt_val_source, 'val', params_transfer, sem_depth=True)

    params_transfer.encoding = params_transfer.encoding_target
    val_dl_target = dataloader.fetch_dataloader(
        args.data_dir, args.txt_val_target, 'val', params_transfer)

    logging.info("- done.")

    # Define the model and optimizer
    model_source = get_network(params_source).to(params_transfer.device)
    model_target = get_network(params_target).to(params_transfer.device)
    transfer = get_transfer(params_transfer).to(params_transfer.device)

    # load source and target model before training and extract backbones
    ckpt_source_file_path = os.path.join(
        args.checkpoint_dir_source, best_ckpt_filename)
    print(ckpt_source_file_path)
    if os.path.exists(ckpt_source_file_path):
        model_source = utils.load_checkpoint(
            model_source, ckpt_dir=args.checkpoint_dir_source, filename=best_ckpt_filename, is_best=True)[0]
        print("=> loaded source model checkpoint form {}".format(
            ckpt_source_file_path))
    else:
        print("=> Initializing source model from scratch")
    ckpt_target_file_path = os.path.join(
        args.checkpoint_dir_target, best_ckpt_filename)
    if os.path.exists(ckpt_target_file_path):
        model_target = utils.load_checkpoint(
            model_target, ckpt_dir=args.checkpoint_dir_target, filename=best_ckpt_filename, is_best=True)[0]
        print("=> loaded target model checkpoint form {}".format(
            ckpt_target_file_path))
    else:
        print("=> Initializing target model from scratch")

    opt1 = optim.AdamW(transfer.parameters(), lr=params_transfer.learning_rate)
    lr_scheduler1 = torch.optim.lr_scheduler.OneCycleLR(
        opt1, max_lr=params_transfer.learning_rate, steps_per_epoch=len(train_dl_all), epochs=params_transfer.num_epochs, div_factor=20)

    opt2 = optim.AdamW(model_source.parameters(), lr=params_source.learning_rate)
    lr_scheduler2 = torch.optim.lr_scheduler.OneCycleLR(
        opt2, max_lr=params_source.learning_rate, steps_per_epoch=len(train_dl_all), epochs=params_transfer.num_epochs, div_factor=20)    

    opt3 = optim.AdamW(model_target.parameters(), lr=params_target.learning_rate)
    lr_scheduler3 = torch.optim.lr_scheduler.OneCycleLR(
        opt3, max_lr=params_target.learning_rate, steps_per_epoch=len(train_dl_all), epochs=params_transfer.num_epochs, div_factor=20)    
        
    # fetch loss function and metrics
    loss_fn1 = get_loss_fn(params_source)
    loss_fn2 = get_loss_fn(params_target)

    # num_classes+1 for background.
    metrics = OrderedDict({})
    for metric in params_transfer.metrics:
        metrics[metric] = get_metrics(metric, params_transfer)

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(
        params_transfer.num_epochs))

    train_and_evaluate(model_source, model_target, transfer, 
                        train_dl_all, train_dl_depth_target, val_dl_source_all, val_dl_target, 
                        opt1, opt2, opt3, loss_fn1, loss_fn2, metrics, params_transfer, 
                        lr_scheduler1, lr_scheduler2, lr_scheduler3, 
                        args.checkpoint_dir_transfer, ckpt_filename, log_dir, writer)