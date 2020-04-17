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

mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

def re_normalize (x, mean=mean, std=std):
    x_r = x.clone()
    for c, (mean_c, std_c) in enumerate(zip(mean, std)):
        x_r[c] *= std_c
        x_r[c] += mean_c
    return x_r

class CarlaDataset(Dataset):
    def __init__(self, root, txt_file, transforms=None, max_depth=1000, threshold=100):
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
        self.transforms = transforms
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
        if self.transforms is not None:
            transformed = self.transforms(image=np.array(img), mask=np.array(gt))
            img = transformed['image']
            gt = transformed['mask']
        img = to_tensor(img)            
        gt = torch.from_numpy(gt)
        
        return img, gt

    def __len__(self):
        return len(self.images)

    def get_predictions_plot(self, batch_sample, predictions, batch_gt):
        num_images = batch_sample.size()[0]
        fig, m_axs = plt.subplots(3, num_images, figsize=(12, 10), squeeze=False)
        plt.subplots_adjust(hspace = 0.1, wspace = 0.1)

        for image, prediction, gt, (axis1, axis2, axis3) in zip(batch_sample, predictions, batch_gt, m_axs.T):
            
            image = re_normalize(image, mean, std)
            image = to_pil_image(image)
            axis1.imshow(image)
            axis1.set_axis_off()

            prediction = self.depth2color(prediction.squeeze())
            axis2.imshow(prediction)
            axis2.set_axis_off()
            
            gt = self.depth2color(gt)
            axis3.imshow(gt)
            axis3.set_axis_off()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches = 'tight', pad_inches = 0)
        buf.seek(0)
        im = Image.open(buf)
        figure = np.array(im)
        buf.close()
        plt.close(fig)
        return figure    

def fetch_dataloader(root, txt_file, split, params, **kwargs):
    h, w = params.crop_h, params.crop_w

    if split == 'train':
        transform_train = Compose([RandomCrop(h,w),
                    HorizontalFlip(p=0.5), 
                    Normalize(mean=mean,std=std)])

        dataset=CarlaDataset(root, txt_file, transforms=transform_train, **kwargs)
        return DataLoader(dataset, batch_size=params.batch_size_train, shuffle=True, num_workers=params.num_workers, drop_last=True, pin_memory=True)

    else:
        transform_val = Compose( [Normalize(mean=mean,std=std)])
        dataset=CarlaDataset(root, txt_file, transforms=transform_val, **kwargs)
        #reduce validation data to speed up training
        if "split_validation" in params.dict:
            ss = ShuffleSplit(n_splits=1, test_size=params.split_validation, random_state=42)
            indexes=range(len(dataset))
            split1, split2 = next(ss.split(indexes))
            dataset=Subset(dataset, split2)        

        return DataLoader(dataset, batch_size=params.batch_size_val, shuffle=False, num_workers=params.num_workers, drop_last=True, pin_memory=True)
