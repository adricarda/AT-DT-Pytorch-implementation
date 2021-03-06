import torch
from sklearn.model_selection import ShuffleSplit
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from albumentations import (
    HorizontalFlip,
    Compose,
    Resize,
    Normalize,
    RandomCrop
)
from dataloader.dataloader_depth import DepthDataset
from dataloader.dataloader_semantic import SegmentationDataset
from dataloader.dataloader_carla_depth_semantic import SemanticDepth


def fetch_dataloader(root, txt_file, split, params, sem_depth=False, txt_file2=None):
    # these can be changed. By deafult we use the target dataset statistics (i.e. Cityscapes)
    mean = [0.286, 0.325, 0.283]
    std = [0.176, 0.180, 0.177]
    h, w = params.crop_h, params.crop_w

    transform_train = Compose([RandomCrop(h, w),
                               HorizontalFlip(p=0.5),
                               Normalize(mean=mean, std=std)], additional_targets={'mask2': 'mask'})
    transform_val = Compose([Normalize(mean=mean, std=std)], additional_targets={'mask2': 'mask'})

    if split == 'train':
        if sem_depth == False:
            if params.task == 'depth':
                dataset = DepthDataset(root, txt_file, transforms=transform_train,
                                       max_depth=params.max_depth, threshold=params.threshold, mean=mean, std=std)
            elif params.task == 'segmentation':
                dataset = SegmentationDataset(
                    root, txt_file, transforms=transform_train, encoding=params.encoding, mean=mean, std=std)
        else:
            dataset = SemanticDepth(root, txt_file, txt_file2, transforms=transform_train, encoding=params.encoding,
                                    max_depth=params.max_depth, threshold=params.threshold, mean=mean, std=std)
        return DataLoader(dataset, batch_size=params.batch_size_train, shuffle=True, num_workers=params.num_workers, pin_memory=True)

    elif split == 'val':
        if sem_depth == False:
            if params.task == 'depth':
                dataset = DepthDataset(root, txt_file, transforms=transform_val,
                                       max_depth=params.max_depth, threshold=params.threshold, mean=mean, std=std)
            elif params.task == 'segmentation':
                dataset = SegmentationDataset(
                    root, txt_file, transforms=transform_val, encoding=params.encoding, mean=mean, std=std)
        else:
            dataset = SemanticDepth(root, txt_file, txt_file2, transforms=transform_val, encoding=params.encoding,
                                    max_depth=params.max_depth, threshold=params.threshold, mean=mean, std=std)

        # reduce validation data to speed up training
        if "split_validation" in params.dict:
            ss = ShuffleSplit(
                n_splits=1, test_size=params.split_validation, random_state=42)
            indexes = range(len(dataset))
            split1, split2 = next(ss.split(indexes))
            dataset = Subset(dataset, split2)

        return DataLoader(dataset, batch_size=params.batch_size_val, shuffle=False, num_workers=params.num_workers, pin_memory=True)
