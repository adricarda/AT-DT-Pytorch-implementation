import random
import os
import io
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor, to_pil_image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.datasets import Cityscapes
from sklearn.model_selection import ShuffleSplit
from torch.utils.data import Subset

from PIL import Image
from albumentations import (
    HorizontalFlip,
    Compose,
    Resize,
    Normalize,
    RandomCrop
    )

from torch.utils.data import Dataset, DataLoader
from collections import namedtuple

mean = [0.286, 0.325, 0.283]
std = [0.176, 0.180, 0.177]

class CarlaDataset(Dataset):
    def __init__(self, root, txt_file, max_depth=1000, threshold=100, transform=None):
        self.files_txt = txt_file
        self.images = []
        self.labels = []
        self.root = root
        for line in open(self.files_txt, 'r').readlines():
            splits = line.split(';')
            self.images.append(os.path.join(root, splits[0].strip()))
            self.labels.append(os.path.join(root, splits[3].strip()))
        self.max_depth = max_depth
        self.treshold = threshold
        self.transform = transform
        self.cm = plt.cm.get_cmap('jet')
        self.colors = self.cm(np.arange(256))[:,:3]

    def png2depth(self, depth_png):
        depth_np = np.array(depth_png, dtype=np.float32)
        depth_np = depth_np[..., 0] + depth_np[..., 1]*256 + depth_np[..., 2]*256*256
        depth_np = depth_np/(256*256*256-1)*self.max_depth
        return np.array(depth_np, dtype=np.float32)

    def depth2color(self, depth_np):
        depth_np[depth_np>self.treshold] = self.treshold
        depth_np = (depth_np-depth_np.min())/(depth_np.max()-depth_np.min())
        indexes = np.array(depth_np*255, dtype=np.int32)
        color_depth = self.colors[indexes]
        return color_depth

    def __getitem__(self, index):
        img = Image.open(self.images[index])          
        gt = Image.open(self.labels[index])
        gt = self.png2depth(gt)
        if self.transform is not None:
            img = self.transform(img)
        img = to_tensor(img)            
        gt = torch.from_numpy(gt)
        
        return img, gt
