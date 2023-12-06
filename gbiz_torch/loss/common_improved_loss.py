# coding:utf-8
# @Author: Haowen Wang

import torch
import torch.nn as nn
from torch.nn import NLLLoss


class LogLoss(object):
    """Wraps torch.nn.NLLLoss to compute loss in gbiz_torch."""

    def __init__(self, dim=-1):
        """
        Args:
            dim: Log soft_max dim

        """
        self.m = nn.LogSoftmax(dim=dim)
        self.loss = nn.NLLLoss()

    def __call__(self, labels, model_output, weights=1.0):
        """
        Args:
            labels: label tensor, shape (batchsize)
            model_output: model output tensors, shape (batchsize, Class_Num)
            weights: sample weights when calculating loss. See
                torch.nn.NLLLoss for more details.
        """
        labels = labels.reshape([-1])
        log_out = self.m(model_output)

        res = self.loss(log_out, labels)

        return res


class MCCrossEntropy(object):
    """
    Wraps torch.nn.MultiLabelSoftMarginLoss to compute loss in gbiz_torch.
    """

    def __init__(self):
        self.loss = nn.MultiLabelSoftMarginLoss()

    def __call__(self, labels, model_output):
        """
        Args:
            labels: (N,C) label targets must have the same shape as the input.
            model_output: (N,C) where N is the batch size and C is the number of classes.
        """
        return self.loss(labels, model_output)


class HuberLoss(object):
    """
    Wraps torch.nn.HuberLoss to compute HuberLoss in gbiz_torch.
    """

    def __init__(self):
        self.loss = nn.HuberLoss()

    def __call__(self, target, input):
        """
        Args:
            target: (*) same shape as the input
            input: (*) where * means any number of dimensions.
        """
        res = self.loss(input, target)
        return res
