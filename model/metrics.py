import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class ConfusionMatrix():
    """Constructs a confusion matrix for a multi-class classification problems.
    Does not support multi-label, multi-class problems.
    Keyword arguments:
    - num_classes (int): number of classes in the classification problem.
    - normalized (boolean, optional): Determines whether or not the confusion
    matrix is normalized or not. Default: False.
    Modified from: https://github.com/pytorch/tnt/blob/master/torchnet/meter/confusionmeter.py
    """

    def __init__(self, num_classes, normalized=False):
        super().__init__()

        self.conf = np.ndarray((num_classes, num_classes), dtype=np.int32)
        self.normalized = normalized
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.conf.fill(0)

    def add(self, predicted, target):
        """Computes the confusion matrix
        The shape of the confusion matrix is K x K, where K is the number
        of classes.
        Keyword arguments:
        - predicted (Tensor or numpy.ndarray): Can be an N x K tensor/array of
        predicted scores obtained from the model for N examples and K classes,
        or an N-tensor/array of integer values between 0 and K-1.
        - target (Tensor or numpy.ndarray): Can be an N x K tensor/array of
        ground-truth classes for N examples and K classes, or an N-tensor/array
        of integer values between 0 and K-1.
        """
        # If target and/or predicted are tensors, convert them to numpy arrays
        if torch.is_tensor(predicted):
            predicted = predicted.cpu().numpy()
        if torch.is_tensor(target):
            target = target.cpu().numpy()

        assert predicted.shape[0] == target.shape[0], \
            'number of targets and predicted outputs do not match'

        if np.ndim(predicted) != 1:
            assert predicted.shape[1] == self.num_classes, \
                'number of predictions does not match size of confusion matrix'
            predicted = np.argmax(predicted, 1)
        else:
            assert (predicted.max() < self.num_classes) and (predicted.min() >= 0), \
                'predicted values are not between 0 and k-1'

        if np.ndim(target) != 1:
            assert target.shape[1] == self.num_classes, \
                'Onehot target does not match size of confusion matrix'
            assert (target >= 0).all() and (target <= 1).all(), \
                'in one-hot encoding, target values should be 0 or 1'
            assert (target.sum(1) == 1).all(), \
                'multi-label setting is not supported'
            target = np.argmax(target, 1)
        else:
            assert (target.max() < self.num_classes) and (target.min() >= 0), \
                'target values are not between 0 and k-1'

        # hack for bincounting 2 arrays together
        x = predicted + self.num_classes * target
        bincount_2d = np.bincount(
            x.astype(np.int32), minlength=self.num_classes**2)
        assert bincount_2d.size == self.num_classes**2
        conf = bincount_2d.reshape((self.num_classes, self.num_classes))

        self.conf += conf

    def value(self):
        """
        Returns:
            Confustion matrix of K rows and K columns, where rows corresponds
            to ground-truth targets and columns corresponds to predicted
            targets.
        """
        if self.normalized:
            conf = self.conf.astype(np.float32)
            return conf / conf.sum(1).clip(min=1e-12)[:, None]
        else:
            return self.conf

class IoU():
    """Computes the intersection over union (IoU) per class and corresponding
    mean (mIoU).
    Intersection over union (IoU) is a common evaluation metric for semantic
    segmentation. The predictions are first accumulated in a confusion matrix
    and the IoU is computed from it as follows:
        IoU = true_positive / (true_positive + false_positive + false_negative).
    Keyword arguments:
    - num_classes (int): number of classes in the classification problem
    - normalized (boolean, optional): Determines whether or not the confusion
    matrix is normalized or not. Default: False.
    - ignore_index (int or iterable, optional): Index of the classes to ignore
    when computing the IoU. Can be an int, or any iterable of ints.
    """

    def __init__(self, num_classes, normalized=False, ignore_index=None):
        super().__init__()
        self.conf_metric = ConfusionMatrix(num_classes, normalized)

        if ignore_index is None:
            self.ignore_index = None
        elif isinstance(ignore_index, int):
            self.ignore_index = (ignore_index,)
        else:
            try:
                self.ignore_index = tuple(ignore_index)
            except TypeError:
                raise ValueError("'ignore_index' must be an int or iterable")

    def reset(self):
        self.conf_metric.reset()

    def add(self, predicted, target):
        """Adds the predicted and target pair to the IoU metric.
        Keyword arguments:
        - predicted (Tensor): Can be a (N, K, H, W) tensor of
        predicted scores obtained from the model for N examples and K classes,
        or (N, H, W) tensor of integer values between 0 and K-1.
        - target (Tensor): Can be a (N, K, H, W) tensor of
        target scores for N examples and K classes, or (N, H, W) tensor of
        integer values between 0 and K-1.
        """
        # Dimensions check
        assert predicted.size(0) == target.size(0), \
            'number of targets and predicted outputs do not match'
        assert predicted.dim() == 3 or predicted.dim() == 4, \
            "predictions must be of dimension (N, H, W) or (N, K, H, W)"
        assert target.dim() == 3 or target.dim() == 4, \
            "targets must be of dimension (N, H, W) or (N, K, H, W)"

        # If the tensor is in categorical format convert it to integer format
        if predicted.dim() == 4:
            _, predicted = predicted.max(1)
        if target.dim() == 4:
            _, target = target.max(1)

        self.conf_metric.add(predicted.view(-1), target.view(-1))

    def value(self):
        """Computes the IoU and mean IoU.
        The mean computation ignores NaN elements of the IoU array.
        Returns:
            Tuple: (IoU, mIoU). The first output is the per class IoU,
            for K classes it's numpy.ndarray with K elements. The second output,
            is the mean IoU.
        """
        conf_matrix = self.conf_metric.value()
        if self.ignore_index is not None:
            for index in self.ignore_index:
                conf_matrix[:, self.ignore_index] = 0
                conf_matrix[self.ignore_index, :] = 0
        true_positive = np.diag(conf_matrix)
        false_positive = np.sum(conf_matrix, 0) - true_positive
        false_negative = np.sum(conf_matrix, 1) - true_positive

        # Just in case we get a division by 0, ignore/hide the error
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = true_positive / (true_positive + false_positive + false_negative)

        return [np.nanmean(iou), iou.tolist()]

class Accuracy():

    def __init__(self, num_classes, normalized=False, ignore_index=None):
        super().__init__()
        self.conf_metric = ConfusionMatrix(num_classes, normalized)

        if ignore_index is None:
            self.ignore_index = None
        elif isinstance(ignore_index, int):
            self.ignore_index = (ignore_index,)
        else:
            try:
                self.ignore_index = tuple(ignore_index)
            except TypeError:
                raise ValueError("'ignore_index' must be an int or iterable")

    def reset(self):
        self.conf_metric.reset()

    def add(self, predicted, target):
        # Dimensions check
        assert predicted.size(0) == target.size(0), \
            'number of targets and predicted outputs do not match'
        assert predicted.dim() == 3 or predicted.dim() == 4, \
            "predictions must be of dimension (N, H, W) or (N, K, H, W)"
        assert target.dim() == 3 or target.dim() == 4, \
            "targets must be of dimension (N, H, W) or (N, K, H, W)"

        # If the tensor is in categorical format convert it to integer format
        if predicted.dim() == 4:
            _, predicted = predicted.max(1)
        if target.dim() == 4:
            _, target = target.max(1)

        self.conf_metric.add(predicted.view(-1), target.view(-1))

    def value(self):

        conf_matrix = self.conf_metric.value()
        if self.ignore_index is not None:
            for index in self.ignore_index:
                conf_matrix[:, self.ignore_index] = 0
                conf_matrix[self.ignore_index, :] = 0

        return [np.diag(conf_matrix).sum() / conf_matrix.sum()]

# class MAE():
#     """
#     Calculates the mean absolute error.
#     """
#     def __init__(self):
#         super().__init__()
#         self._sum_of_absolute_errors = 0.0
#         self._num_examples = 0

#     def reset(self):
#         self._sum_of_absolute_errors = 0.0
#         self._num_examples = 0

#     def add(self, predicted, target):
#                 # If target and/or predicted are tensors, convert them to numpy arrays
#         if torch.is_tensor(predicted):
#             predicted = predicted.cpu().numpy()
#         if torch.is_tensor(target):
#             target = target.cpu().numpy()
        
#         assert predicted.shape[0] == target.shape[0], \
#             'number of targets and predicted outputs do not match'
#         if np.ndim(target) != np.ndim(predicted):
#             predicted = predicted.squeeze(axis=1)
#             assert target.shape == predicted.shape, \
#             'target and predicted shapes do not match'

#         #rescale prediction from [0,1] to [0, max_depth] (labels are in [0, max_depth] too)
#         predicted = predicted*self.max_depth
#         predicted[predicted<self.min_depth] = self.min_depth
#         predicted[predicted>self.max_depth] = self.max_depth
#         mask = np.logical_and(target > self.min_depth, target < self.max_depth)

#         absolute_errors = np.abs(predicted[mask] - target[mask])
#         self._sum_of_absolute_errors += np.sum(absolute_errors).item()
#         self._num_examples += target.shape[0]

#     def value(self):
#         if self._num_examples == 0:
#             raise ZeroDivisionError('MeanAbsoluteError must have at least one example before it can be computed.')
#         return [self._sum_of_absolute_errors / self._num_examples]

# class MeanSquaredError():
#     """
#     Calculates the mean squared error.
#     """
#     def __init__(self, min_depth=0.001, max_depth=100):
#         super().__init__()
#         self._sum_of_absolute_errors = 0.0
#         self._num_examples = 0
#         self.min_depth = min_depth
#         self.max_depth = max_depth

#     def reset(self):
#         self._sum_of_absolute_errors = 0.0
#         self._num_examples = 0

#     def add(self, predicted, target):
#         if torch.is_tensor(predicted):
#             predicted = predicted.cpu().numpy()
#         if torch.is_tensor(target):
#             target = target.cpu().numpy()

#         assert predicted.shape[0] == target.shape[0], \
#             'number of targets and predicted outputs do not match'
#         if np.ndim(target) != np.ndim(predicted):
#             predicted = predicted.squeeze(axis=1)
#             assert target.shape == predicted.shape, \
#             'target and predicted shapes do not match'

#         #rescale prediction from [0,1] to [0, max_depth] (labels are in [0, max_depth] too)
#         predicted = predicted*self.max_depth
#         predicted[predicted<self.min_depth] = self.min_depth
#         predicted[predicted>self.max_depth] = self.max_depth
#         mask = np.logical_and(target > self.min_depth, target < self.max_depth)

#         squared_errors = np.power(predicted[mask] - target[mask], 2)
#         self._sum_of_squared_errors += np.sum(squared_errors).item()
#         self._num_examples += target.shape[0]

#     def value(self):
#         if self._num_examples == 0:
#             raise ZeroDivisionError('MeanSquaredError must have at least one example before it can be computed.')
#         return [self._sum_of_squared_errors / self._num_examples]

# class RootMeanSquaredError(MeanSquaredError):
#     """
#     Calculates the root mean squared error.
#     """
#     def value(self):
#         mse = super(RootMeanSquaredError, self).value()
#         return [math.sqrt(mse[0])]


class RMSE(nn.Module):
    def __init__(self):
        super(RMSE, self).__init__()
        self.errors = 0
        self._num_examples = 0
        self.max_depth = 100
        self.min_depth = 0.001
    
    def reset(self):
        self.errors = 0
        self._num_examples = 0        
    
    def add(self, predicted, target):

        predicted = predicted.squeeze(dim=1)
        predicted = predicted*self.max_depth
        predicted[predicted<self.min_depth] = self.min_depth
        predicted[predicted>self.max_depth] = self.max_depth
        mask = (target > self.min_depth) & (target < self.max_depth)
        predicted[~mask] = 0
        target[~mask] = 0

        norms = torch.sum(torch.abs(predicted - target)**2, dim=(1,2))/torch.sum(mask, dim=(1,2))
        norms = torch.sqrt(norms)
        self._num_examples += predicted.size()[0]

    def value(self):
      error = self.errors / self._num_examples
      return [error.item()]

# class LogRootMeanSquaredError(RootMeanSquaredError):
#     """
#     Calculates the log root mean squared error.
#     """
#     def add(self, predicted, target):
#         if torch.is_tensor(predicted):
#             predicted = predicted.cpu().numpy()
#         if torch.is_tensor(target):
#             target = target.cpu().numpy()

#         assert predicted.shape[0] == target.shape[0], \
#             'number of targets and predicted outputs do not match'
#         if np.ndim(target) != np.ndim(predicted):
#             predicted = predicted.squeeze(axis=1)
#             assert target.shape == predicted.shape, \
#             'target and predicted shapes do not match'

#         #rescale prediction from [0,1] to [0, max_depth] (labels are in [0, max_depth] too)
#         predicted = predicted*self.max_depth
#         predicted[predicted<self.min_depth] = self.min_depth
#         predicted[predicted>self.max_depth] = self.max_depth
#         mask = np.logical_and(target > self.min_depth, target < self.max_depth)
#         squared_errors = np.power(np.log(predicted[mask]) - np.log(target[mask]), 2)
#         self._sum_of_squared_errors += np.sum(squared_errors).item()
#         self._num_examples += target.shape[0]


def get_metrics(metrics_name="iou", **kwargs):
    if metrics_name=='iou':
        return IoU(**kwargs)
    if metrics_name=='accuracy':
        return Accuracy(**kwargs)
    if metrics_name=='mse':
        return MeanSquaredError(**kwargs)        
    if metrics_name=='rmse':
        return RMSE(**kwargs)
    if metrics_name=='log_rmse':
        return LogRootMeanSquaredError(**kwargs)