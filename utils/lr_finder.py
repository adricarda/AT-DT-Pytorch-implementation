import torch
import math
import argparse
import os
import random
import utils.utils as utils
import numpy as np
import matplotlib.pyplot as plt
import dataloader.dataloader as data_loader
from model.net import get_network
from model.losses import get_loss_fn
import torch.optim as optim
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/baseline',
                    help="Directory containing params.json")
parser.add_argument('--checkpoint_dir', default=None,
                    help="Directory containing weights to reload before \
                    training")
parser.add_argument('--txt_train', default='/content/drive/My Drive/atdt/input_list_train_carla.txt',
                    help="Txt file containing path to training images")
parser.add_argument('--txt_val', default='/content/drive/My Drive/atdt/input_list_val_carla.txt',
                    help="Txt file containing path to validation images")                    

def find_lr(data_ld, opt, model, criterion, device, init_value = 1e-8, final_value=10., beta = 0.98):
    num = len(data_ld)-1
    mult = (final_value / init_value) ** (1/num)
    lr = init_value
    opt.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []
    for (inputs,labels) in tqdm(data_ld):
        batch_num += 1
        #As before, get the loss for this mini-batch of inputs/outputs
        inputs, labels = inputs.to(device), labels.to(device)
        opt.zero_grad()
        outputs = model(inputs)['out']
        loss = criterion(outputs, labels)
        #Compute the smoothed loss
        avg_loss = beta * avg_loss + (1-beta) *loss.item()
        smoothed_loss = avg_loss / (1 - beta**batch_num)
        #Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 3 * best_loss:
            return log_lrs, losses
        #Record the best loss
        if smoothed_loss < best_loss or batch_num==1:
            best_loss = smoothed_loss
        #Store the values
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))
        #Do the SGD step
        loss.backward()
        opt.step()
        #Update the lr for the next step
        lr *= mult
        opt.param_groups[0]['lr'] = lr
    return log_lrs, losses

def plot_lr(log_lrs, losses):
    plt.plot(log_lrs[10:-5],losses[10:-5])
    plt.savefig('lr_plt.png')
    

if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Set the random seed for reproducible experiments
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    train_dl = data_loader.fetch_dataloader(args.data_dir, args.txt_train, 'train', params)

    # Define the model and optimizer
    model = get_network(params).to(device)
    opt = optim.AdamW(model.parameters(), lr=params.learning_rate)
    loss_fn = get_loss_fn(loss_name=params.loss_fn)

    if args.checkpoint_dir:
        model = utils.load_checkpoint(model, is_best=False, ckpt_dir=args.checkpoint_dir)[0]
    
    log_lrs, losses = find_lr(train_dl, opt, model, loss_fn, device)
    plot_lr(log_lrs, losses)
