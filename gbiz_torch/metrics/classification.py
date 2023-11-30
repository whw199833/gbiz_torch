# coding:utf-8
# @Author: Haowen Wang

import torch
import sklearn.metrics as sm


class AUC(object):
    """Wraps sklearn.metrics.roc_auc_score to calculate auc in gbiz_torch."""

    def __init__(self):
        self.auc = sm.roc_auc_score

    def __call__(self, labels, model_output):
        """
        Args:
            labels: label tensor
            model_output: model predict tensor

        """
        return self.auc(labels, model_output)
