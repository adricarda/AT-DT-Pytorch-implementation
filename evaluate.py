"""Evaluates the model"""

import argparse
import logging
import os
import random
import numpy as np
import torch
import utils
from model.net import get_network
from tqdm import tqdm
import dataloader.dataloader_depth as dataloader_depth
from model.losses import get_loss_fn
from model.metrics import get_metrics

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/baseline',
                    help="Directory containing params.json")
parser.add_argument('--checkpoint_dir', default="experiments/baseline/checkpoints",
                    help="Directory containing weights to reload before \
                    training") 


def evaluate(model,loss_fn,dataset_dl,opt=None, metrics=None, params=None):

    # set model to evaluation mode
    model.eval()
    running_loss=utils.RunningAverage()
    num_batches=len(dataset_dl)
    if metrics is not None:
        for metric_name, metric in metrics.items(): 
            metric.reset()

    with torch.no_grad():        
        for (xb, yb) in tqdm(dataset_dl):
            xb=xb.to(params.device)
            yb=yb.to(params.device)    
            output=model(xb)['out']

            loss_b = loss_fn(output, yb)
            running_loss.update(loss_b.item())
            if metrics is not None:            
                # output=torch.argmax(output.detach(), dim=1)
                for metric_name, metric in metrics.items(): 
                    metric.add(output, yb)

    if metrics is not None:
        metrics_results = {}
        for metric_name, metric in metrics.items(): 
            metrics_results[metric_name] = metric.value()                  
        return running_loss(), metrics_results
    else:   
        return running_loss(), None

if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    ckpt_filename = "checkpoint.tar"
    # use GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    params.device = device

    # Set the random seed for reproducible experiments
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # fetch dataloaders
    val_dl = dataloader_depth.fetch_dataloader(args.data_dir, 'val', params)

    # Define the model
    model = get_network(params).to(params.device)

    # fetch loss function and metrics
    loss_fn = get_loss_fn(loss_name=params.loss_fn , ignore_index=19)
    #num_classes+1 for background.
    metrics = {}
    for metric in params.metrics:
        metrics[metric]= get_metrics(metrics_name=metric,
                num_classes=params.num_classes+1, ignore_index=params.ignore_index)

    # Reload weights from the saved file
    model, _, _, _, _ = utils.load_checkpoint(model, is_best=True, checkpoint_dir=args.checkpoint_dir)

    # Evaluate
    eval_loss, val_metrics = evaluate(model, loss_fn, val_dl, metric=metrics, params=params)
    best_json_path = os.path.join(args.model_dir, "evaluation.json")
    for val_metric_name, val_metric_results in val_metrics.items(): 
        logging.info("%s: %.3f" %(val_metric_name, val_metric_results))
    utils.save_dict_to_json(val_metrics, best_json_path)      