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

parser.add_argument('--model_dir_source', default='experiments/end2end/source',
                    help="Directory containing source params.json")
parser.add_argument('--model_dir_target', default='experiments/end2end/target',
                    help="Directory containing target params.json")
parser.add_argument('--model_dir_transfer', default='experiments/end2end/transfer',
                    help="Directory containing transfer params.json")

parser.add_argument('--checkpoint_dir_source', default="experiments/end2end/source/ckpt",
                    help="Directory containing source model weights")
parser.add_argument('--checkpoint_dir_target', default="experiments/end2end/target/ckpt",
                    help="Directory containing weights target model weights")
parser.add_argument('--checkpoint_dir_transfer', default="experiments/end2end/transfer/ckpt",
                    help="Directory containing weights target model weights")

parser.add_argument('--tensorboard_dir', default="experiments/end2end/tensorboard",
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


def evaluate_source(model_source, model_target, dataset_dl, metrics_depth, metrics_segmentation, params):
    
    # set model to evaluation mode
    model_source.eval()
    model_target.eval()
    metrics_depth_results = {}
    metrics_segmentation_results = {}

    if metrics_depth is not None:
        for metric_name, metric in metrics_depth.items():
            metric.reset()
    if metrics_segmentation is not None:
        for metric_name, metric in metrics_segmentation.items():
            metric.reset()

    with torch.no_grad():
        for (xb, yb_seg, yb_depth, _, _) in tqdm(dataset_dl):
            xb = xb.to(params.device)
            yb_seg = yb_seg.to(params.device)
            yb_depth = yb_depth.to(params.device)

            output_depth = model_source(xb)['out']
            output_seg = model_target(xb)['out']

            if metrics_depth is not None:
                for metric_name, metric in metrics_depth.items():
                    metric.add(output_depth, yb_depth)
            if metrics_segmentation is not None:
                for metric_name, metric in metrics_segmentation.items():
                    metric.add(output_seg, yb_seg)

    if metrics_depth is not None:
        for metric_name, metric in metrics_depth.items():
            metrics_depth_results[metric_name] = metric.value()
    if metrics_segmentation is not None:
        for metric_name, metric in metrics_segmentation.items():
            metrics_segmentation_results[metric_name] = metric.value()

    return metrics_depth_results, metrics_segmentation_results


class LayerActivationHook:
    features = None

    def __init__(self, model):
        self.hook = model.backbone.register_forward_hook(self.hook_fn)
    
    def hook_fn(self, model, input, output):
        self.features = output['out'].detach()
    
    def remove_hook(self):
        self.hook.remove()


def train_epoch(model_source, model_target, transfer, train_dl_all, 
                opt1, opt2, opt3, loss_fn1, loss_fn2, params,
                lr_scheduler1, lr_scheduler2, lr_scheduler3):

    running_loss_depth_carla = utils.RunningAverage()
    running_loss_segmentation_carla = utils.RunningAverage()

    source_encoder = model_source.backbone
    source_decoder = model_source.classifier
    target_decoder = model_target.classifier

    for (batch_images_carla, batch_segmentation_carla, batch_depth_carla, batch_images_cs, batch_depth_cs) in tqdm(train_dl_all):
        input_shape = batch_images_carla.shape[-2:]

        hook = LayerActivationHook(model_source)
        # try:
        #     batch_images_cs, batch_depth_cs = next(iter_cs_depth)
        # except StopIteration:
        #     iter_cs_depth = iter(train_dl_depth_target)
        #     batch_images_cs, batch_depth_cs = next(iter_cs_depth)

        loss_cs_depth = train_step(model_source, batch_images_cs.to(params.device), batch_depth_cs.to(params.device), opt1, loss_fn1)

        batch_images_carla = batch_images_carla.to(params.device)
        batch_segmentation_carla = batch_segmentation_carla.to(params.device)
        batch_depth_carla = batch_depth_carla.to(params.device)

        # depth_feature = source_encoder(batch_images_carla)['out']
        # depth_feature_copy = depth_feature.detach()
        hook.features = None
        depth_prediction = model_source(batch_images_carla)
        depth_feature_copy = hook.features
        hook.remove_hook()

        depth_prediction = F.interpolate(depth_prediction['out'], size=input_shape, mode='bilinear', align_corners=False)
        loss_depth_carla = loss_fn1(depth_prediction, batch_depth_carla)

        if opt1 is not None:
            opt1.zero_grad()
            loss_depth_carla.backward()
            opt1.step()

        if lr_scheduler1 is not None:
            lr_scheduler1.step()       

        loss_segmentation_carla = train_step(model_target, batch_images_carla, batch_segmentation_carla, opt2, loss_fn2)
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
        running_loss_segmentation_carla.update(loss_segmentation_carla.item())

    return running_loss_depth_carla(), running_loss_segmentation_carla()


def train_and_evaluate(model_source, model_target, transfer, train_dl_all, val_dl_all, val_dl_target, 
                        opt1, opt2, opt3, loss_fn1, loss_fn2, metrics_depth, metrics_segmentation, params,
                        lr_scheduler1, lr_scheduler2, lr_scheduler3,
                        checkpoint_dir_source, checkpoint_dir_target, checkpoint_dir_transfer, 
                        ckpt_filename, log_dir, writer):

    ckpt_file_path = os.path.join(checkpoint_dir_transfer, ckpt_filename)
    best_value = -float('inf')
    start_epoch = 0

    batch_sample_carla, batch_gt_carla_sem, batch_gt_carla_depth, _, _ = next(iter(val_dl_all))
    batch_sample_cs, batch_gt_cs = next(iter(val_dl_target))

    if os.path.exists(ckpt_file_path):
        transfer, opt3, lr_scheduler3, start_epoch, best_value = utils.load_checkpoint(transfer, opt3, lr_scheduler3,
                                                                start_epoch, False, best_value, checkpoint_dir_transfer, ckpt_filename)
        print("=> loaded transfer checkpoint form {} (epoch {})".format(
            ckpt_file_path, start_epoch))
    else:
        print("=> Initializing transfer from scratch")

    source_encoder = model_source.backbone
    target_decoder = model_target.classifier
    adpative_model = get_adaptive_network(source_encoder, transfer, target_decoder)

    for epoch in range(start_epoch, params.num_epochs):
        # Run one epoch
        current_lr = get_lr(opt3)
        logging.info('Epoch {}/{}, current lr={}'.format(epoch,
                                                         params.num_epochs-1, current_lr))
        writer.add_scalar('Learning_rate', current_lr, epoch)

        transfer.train()
        train_loss_depth, train_loss_segmentation = train_epoch(
                        model_source, model_target, transfer, train_dl_all, 
                        opt1, opt2, opt3, loss_fn1, loss_fn2, params,
                        lr_scheduler1, lr_scheduler2, lr_scheduler3)

        writer.add_scalars('Losses', {
            'Training_depth': train_loss_depth,
            'Training_segmentation': train_loss_segmentation,
        }, epoch)

        # if epoch % 5 == 0 or epoch==params.num_epochs-1:
        predictions_sem = inference(model_target, batch_sample_carla)
        predictions_depth = inference(model_source, batch_sample_carla)

        plot = train_dl_all.dataset.get_predictions_plot(
            batch_sample_carla, predictions_sem.cpu(), batch_gt_carla_sem.cpu(), predictions_depth.cpu(), batch_gt_carla_depth.cpu())
        writer.add_image('Predictions_carla', plot, epoch, dataformats='HWC')

        predictions = inference(adpative_model, batch_sample_cs)
        plot = val_dl_target.dataset.dataset.get_predictions_plot(
            batch_sample_cs, predictions.cpu(), batch_gt_cs)
        writer.add_image('Predictions_target', plot, epoch, dataformats='HWC')

        val_metrics_depth, val_metrics_segmentation = evaluate_source(
        model_source, model_target, val_dl_all, metrics_depth, metrics_segmentation, params)

        _, val_metrics_transfer = evaluate(
            adpative_model, None, val_dl_target, metrics=metrics_segmentation, params=params)

        for (val_metric_name, val_metric_results) in val_metrics_depth.items():
            writer.add_scalar(val_metric_name, val_metric_results[0], epoch)

        for (val_metric_name, val_metric_results) in val_metrics_segmentation.items():
            writer.add_scalar(val_metric_name+'_target', val_metric_results[0], epoch)

        for (val_metric_name, val_metric_results) in val_metrics_transfer.items():
            writer.add_scalar(val_metric_name+'_transfer', val_metric_results[0], epoch)

        current_value = list(val_metrics_transfer.values())[0][0]
        is_best = current_value >= best_value

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best value")
            best_value = current_value
            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(
                log_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics_transfer, best_json_path)

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model_source.state_dict(),
                               'optim_dict': opt1.state_dict(),
                               'scheduler_dict': lr_scheduler1.state_dict(),
                               'best_value': best_value},
                              is_best=is_best,
                              ckpt_dir=checkpoint_dir_source,
                              filename=ckpt_filename)

        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model_target.state_dict(),
                               'optim_dict': opt2.state_dict(),
                               'scheduler_dict': lr_scheduler2.state_dict(),
                               'best_value': best_value},
                              is_best=is_best,
                              ckpt_dir=checkpoint_dir_target,
                              filename=ckpt_filename)

        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': transfer.state_dict(),
                               'optim_dict': opt3.state_dict(),
                               'scheduler_dict': lr_scheduler3.state_dict(),
                               'best_value': best_value},
                              is_best=is_best,
                              ckpt_dir=checkpoint_dir_transfer,
                              filename=ckpt_filename)                              

        logging.info("\ntrain loss depth: %.3f, train loss segmentation: %.3f" %
                     (train_loss_depth, train_loss_segmentation))

        for (val_metric_name, val_metric_results) in val_metrics_depth.items():
            logging.info("val depth %s: %.3f" % (val_metric_name, val_metric_results[0]))

        for (val_metric_name, val_metric_results) in val_metrics_segmentation.items():
            logging.info("val segmentation target %s: %.3f" % (val_metric_name, val_metric_results[0]))

        for (val_metric_name, val_metric_results) in val_metrics_transfer.items():
            logging.info("val segmentation transfer %s: %.3f" % (val_metric_name, val_metric_results[0]))

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
        args.data_dir, args.txt_train_carla, 'train', params_transfer, sem_depth=True, txt_file2=args.txt_train_cs)

    # train_dl_depth_target = dataloader.fetch_dataloader(
    #     args.data_dir, args.txt_train_cs, 'train', params_source)

    val_dl_all = dataloader.fetch_dataloader(
        args.data_dir, args.txt_val_source, 'val', params_transfer, sem_depth=True, txt_file2=args.txt_val_target)

    params_transfer.encoding = params_transfer.encoding_target
    val_dl_target = dataloader.fetch_dataloader(
        args.data_dir, args.txt_val_target, 'val', params_transfer)

    logging.info("- done.")

    # Define the model and optimizer
    model_source = get_network(params_source).to(params_transfer.device)
    model_target = get_network(params_target).to(params_transfer.device)
    transfer = get_transfer(params_transfer).to(params_transfer.device)

    opt1 = optim.AdamW(model_source.parameters(), lr=params_source.learning_rate)
    lr_scheduler1 = torch.optim.lr_scheduler.OneCycleLR(
        opt1, max_lr=params_source.learning_rate, steps_per_epoch=len(train_dl_all), epochs=params_transfer.num_epochs, div_factor=20)    

    opt2 = optim.AdamW(model_target.parameters(), lr=params_target.learning_rate)
    lr_scheduler2 = torch.optim.lr_scheduler.OneCycleLR(
        opt2, max_lr=params_target.learning_rate, steps_per_epoch=len(train_dl_all), epochs=params_transfer.num_epochs, div_factor=20)    

    opt3 = optim.AdamW(transfer.parameters(), lr=params_transfer.learning_rate)
    lr_scheduler3 = torch.optim.lr_scheduler.OneCycleLR(
        opt3, max_lr=params_transfer.learning_rate, steps_per_epoch=len(train_dl_all), epochs=params_transfer.num_epochs, div_factor=20)

    # load source and target model before training and extract backbones
    ckpt_source_file_path = os.path.join(
        args.checkpoint_dir_source, ckpt_filename)
    if os.path.exists(ckpt_source_file_path):
        model_source, _, _, _, _ = utils.load_checkpoint(
            model_source, None, None, ckpt_dir=args.checkpoint_dir_source, filename=ckpt_filename)
        print("=> loaded source model checkpoint form {}".format(
            ckpt_source_file_path))
    else:
        print("=> Initializing source model from scratch")

    ckpt_target_file_path = os.path.join(
        args.checkpoint_dir_target, ckpt_filename)
    if os.path.exists(ckpt_target_file_path):
        model_target, _, _, _, _ = utils.load_checkpoint(
            model_target, None, None, ckpt_dir=args.checkpoint_dir_target, filename=ckpt_filename)
        print("=> loaded target model checkpoint form {}".format(
            ckpt_target_file_path))
    else:
        print("=> Initializing target model from scratch")

    # fetch loss function and metrics
    loss_fn1 = get_loss_fn(params_source)
    loss_fn2 = get_loss_fn(params_target)

    metrics_depth = OrderedDict({})
    for metric in params_source.metrics:
        metrics_depth[metric] = get_metrics(metric, params_source)

    metrics_segmentation = OrderedDict({})
    for metric in params_target.metrics:
        metrics_segmentation[metric] = get_metrics(metric, params_target)

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(
        params_transfer.num_epochs))

    train_and_evaluate(model_source, model_target, transfer, 
                        train_dl_all, val_dl_all, val_dl_target, 
                        opt1, opt2, opt3, loss_fn1, loss_fn2, metrics_depth, metrics_segmentation, params_transfer, 
                        lr_scheduler1, lr_scheduler2, lr_scheduler3, 
                        args.checkpoint_dir_source, args.checkpoint_dir_target, args.checkpoint_dir_transfer, 
                        ckpt_filename, log_dir, writer)