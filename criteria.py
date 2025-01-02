# -*- coding: utf-8 -*-
# @Time    : 2018/10/23 20:04
# @Author  : Wang Xin
# @Email   : wangxin_buaa@163.com


import torch
import torch.nn as nn


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = (diff ** 2).mean()
        return self.loss


class MaskedL1Loss(nn.Module):
    def __init__(self, upper_limit=1.0e6):
        super(MaskedL1Loss, self).__init__()
        self.upper_limit = upper_limit

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = ((target > 0) & (target < self.upper_limit)).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss


class berHuLoss(nn.Module):
    def __init__(self, upper_limit=1.0e6):
        super(berHuLoss, self).__init__()
        self.upper_limit = upper_limit

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"

        target = torch.clamp(target, 0.0, self.upper_limit)
        valid_mask = (target > 0).detach()

        diff = target - pred
        diff = diff[valid_mask]
        diff = diff.abs()
        
        huber_c = torch.max(diff)
        huber_c = 0.2 * huber_c 

        huber_mask = (diff > huber_c).detach()
        diff_1 = diff[~huber_mask]
        diff_2 = (diff[huber_mask]**2 + huber_c**2) / (2*huber_c)
                
        self.loss = torch.cat((diff_1, diff_2)).mean()

        return self.loss
