# coding:utf-8
# @Author: Haowen Wang

import torch
import torch.nn as nn


class CoxRegressionLoss(object):
    def __init__(self):
        self.predict_key = 'prediction'

    def __call__(self, labels, model_outputs):
        """
        labels: binary input of {0, 1} in shape (batch_size, 1)
        model_outputs: probability in range (0, 1) in shape (batch_size, 1)
        """
        logits = model_outputs
        logits = logits.reshape(-1, 1)
        labels = labels.reshape(-1, 1)

        binary_labels = torch.cat([labels, 1 - labels], axis=1)
        y_ = binary_labels.to(torch.float32)
        y = torch.concat([-torch.log(1.0 - torch.exp(-logits)), logits], 1)
        loss = torch.mean(torch.multiply(y, y_)) * 2.0
        return loss
