import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class BerHu(nn.Module):
    def __init__(self, meter_threshold=100, error_threshold=0.2):
        super(BerHu, self).__init__()
        self.error_threshold = error_threshold
        self.meter_threshold = meter_threshold
    
    def forward(self, real, fake):
        mask = real>0
        real[real>self.meter_threshold] = self.meter_threshold
        real /= self.meter_threshold

        if not fake.shape == real.shape:
            _,_,H,W = real.shape
            fake = F.upsample(fake, size=(H,W), mode='bilinear')
        fake = fake * mask
        real = real * mask

        diff = torch.abs(real-fake)
        delta = self.error_threshold * torch.max(diff).item()

        part1 = -F.error_threshold(-diff, -delta, 0.)
        part2 = F.error_threshold(diff**2 - delta**2, 0., -delta**2.) + delta**2
        part2 = part2 / (2.*delta)

        loss = part1 + part2
        loss = torch.sum(loss)
        return loss

def get_loss_fn(loss_name='crossentropy', **kwargs):

    if loss_name=='crossentropy':
        return nn.CrossEntropyLoss(**kwargs)
    if loss_name=='beruh':
        return nn.BerHu(**kwargs)
    else:
        return nn.CrossEntropyLoss(**kwargs)