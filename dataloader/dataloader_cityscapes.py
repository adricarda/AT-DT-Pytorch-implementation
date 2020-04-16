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

# Based on https://github.com/mcordts/cityscapesScripts
CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                  'has_instances', 'ignore_in_eval', 'color'])

classes = [
    CityscapesClass('unlabeled', 0, 19, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('ego vehicle', 1, 19, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('rectification border', 2, 19, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('out of roi', 3, 19, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('static', 4, 19, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('dynamic', 5, 19, 'void', 0, False, True, (111, 74, 0)),
    CityscapesClass('ground', 6, 19, 'void', 0, False, True, (81, 0, 81)),
    CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
    CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
    CityscapesClass('parking', 9, 19, 'flat', 1, False, True, (250, 170, 160)),
    CityscapesClass('rail track', 10, 19, 'flat', 1, False, True, (230, 150, 140)),
    CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
    CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
    CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
    CityscapesClass('guard rail', 14, 19, 'construction', 2, False, True, (180, 165, 180)),
    CityscapesClass('bridge', 15, 19, 'construction', 2, False, True, (150, 100, 100)),
    CityscapesClass('tunnel', 16, 19, 'construction', 2, False, True, (150, 120, 90)),
    CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
    CityscapesClass('polegroup', 18, 19, 'object', 3, False, True, (153, 153, 153)),
    CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
    CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
    CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
    CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
    CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
    CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
    CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
    CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
    CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    CityscapesClass('caravan', 29, 19, 'vehicle', 7, True, True, (0, 0, 90)),
    CityscapesClass('trailer', 30, 19, 'vehicle', 7, True, True, (0, 0, 110)),
    CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
    CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
    CityscapesClass('license plate', -1, 19, 'vehicle', 7, False, True, (0, 0, 142)),
]

palette = []
labels = range(19)
for cs_class in classes:
  if cs_class.train_id in labels:
    R, G, B = cs_class.color
    palette.extend((R, G, B))

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

class CityScapesDataset(Cityscapes):
    def __init__(self, root, split='train', mode='fine', target_type='instance',
                 transform=None, target_transform=None, transforms=None, ignore_label=19, classes=None):
    
        self.classes = classes
        super(CityScapesDataset, self).__init__(root, split, mode, target_type,
                 transform, target_transform, transforms)

        self.id_to_trainId = {cs_class.id: cs_class.train_id for cs_class in self.classes}

    def __getitem__(self, index):
        img_path, mask_path = self.images[index], self.targets[index][0]
        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)

        mask = np.array(mask)
        mask_copy = mask.copy()
        for k, v in self.id_to_trainId.items():
            mask_copy[mask == k] = v
        mask = Image.fromarray(mask_copy.astype(np.uint8))

        if self.transforms is not None:
            transformed = self.transforms(image=np.array(img), mask=np.array(mask))
            img = transformed['image']
            mask = transformed['mask']
        img = to_tensor(img)            
        mask = torch.from_numpy(np.array(mask)).type(torch.long)    
        return img, mask


def fetch_dataloader(data_dir, split, params):
    h, w = params.crop_h, params.crop_w

    if split == 'train':
        transform_train = Compose([RandomCrop(h,w),
                    HorizontalFlip(p=0.5), 
                    Normalize(mean=mean,std=std)])

        dataset=CityScapesDataset(data_dir, split=split, mode='fine',
                    target_type='semantic', transforms=transform_train, classes=classes)
        return DataLoader(dataset, batch_size=params.batch_size_train, shuffle=True, num_workers=params.num_workers, drop_last=True, pin_memory=True)

    else:
        transform_val = Compose( [Normalize(mean=mean,std=std)])
        dataset=CityScapesDataset(data_dir, split=split, mode='fine',
                    target_type='semantic', transforms=transform_val, classes=classes)
        #reduce validation data to speed up training
        if "split_validation" in params.dict:
            ss = ShuffleSplit(n_splits=1, test_size=params.split_validation, random_state=42)
            indexes=range(len(dataset))
            split1, split2 = next(ss.split(indexes))
            dataset=Subset(dataset, split2)        

        return DataLoader(dataset, batch_size=params.batch_size_val, shuffle=False, num_workers=params.num_workers, drop_last=True, pin_memory=True)


def colorize_mask(mask, palette):
    # mask: numpy array of the mask
    mask = np.array(mask)
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

def re_normalize (x, mean=mean, std=std):
    x_r = x.clone()
    for c, (mean_c, std_c) in enumerate(zip(mean, std)):
        x_r[c] *= std_c
        x_r[c] += mean_c
    return x_r

def get_predictions_plot(batch_sample, predictions, batch_gt):

    num_images = batch_sample.size()[0]
    fig, m_axs = plt.subplots(3, num_images, figsize=(12, 10), squeeze=False)
    plt.subplots_adjust(hspace = 0.1, wspace = 0.1)

    for image, prediction, gt, (axis1, axis2, axis3) in zip(batch_sample, predictions, batch_gt, m_axs.T):
        
        image = re_normalize(image, mean, std)
        image = to_pil_image(image)
        axis1.imshow(image)
        axis1.set_axis_off()

        prediction = colorize_mask(prediction, palette)
        axis2.imshow(prediction)
        axis2.set_axis_off()
        
        gt = colorize_mask(gt, palette)
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

