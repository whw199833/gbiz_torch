# coding:utf-8
# @Author: Haowen Wang

import torch
import sklearn.metrics as sm


class r2_score(object):
    """
        Wraps sklearn.metrics.r2_score to calculate r2_score in gbiz_torch.
    """

    def __init__(self):
        self.r2_score = sm.r2_score

    def __call__(self, y_label, y_pred):
        """
          y_label: array-like of shape (n_samples,) or (n_samples, n_outputs)
                    Ground truth (correct) target values.
          y_pred: array-like of shape (n_samples,) or (n_samples, n_outputs)
                    Estimated target values.
        """
        res = self.r2_score(y_label, y_pred)
        return res


class MME(object):
    """
        Wraps sklearn.metrics.{mean_absolute_error, mean_squared_error, mean_absolute_percentage_error} to calculate MAE, MSE, MAPE in gbiz_torch.
    """

    def __init__(self):
        self.mae = sm.mean_absolute_error
        self.mse = sm.mean_squared_error
        self.mape = sm.mean_absolute_percentage_error

    def __call__(self, y_label, y_pred):
        """
          y_label: array-like of shape (n_samples,) or (n_samples, n_outputs)
                    Ground truth (correct) target values.
          y_pred: array-like of shape (n_samples,) or (n_samples, n_outputs)
                    Estimated target values.
        """
        res = {'MAE': self.mae(y_label, y_pred), 'MSE': self.mse(
            y_label, y_pred), 'MAPE': self.mape(y_label, y_pred)}
        return res
