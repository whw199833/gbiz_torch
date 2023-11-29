# coding: utf-8
# @Author: Haowen Wang

import torch
import torch.nn as nn
import torch.nn.functional as F
from gbiz_torch.layer import DNNLayer, CrossLayer


class DCNModel(nn.Module):
    """
      Model: DCN Model

      Paper: Deep & Cross Network for Ad Click Predictions

      Link: https://arxiv.org/abs/1708.05123

      Author: Ruoxi Wang, Bin Fu, Gang Fu, Mingliang Wang

      Developer: Haowen Wang

      inputs: 2d tensor (batch_size, n_dim)

      outputs: 2d tensor (batch_size, out_dim)

      """

    def __init__(self,
                 in_shape,
                 hidden_units,
                 act_fn='relu',
                 l2_reg=0.001,
                 dropout_rate=0,
                 use_bn=False,
                 n_cross_layers=2,
                 seed=1024):
        """
            Args:
                hidden_units: list, unit in each hidden layer
                act_fn: string, activation function
                l2_reg: float, regularization value
                dropout_rate: float, fraction of the units to dropout.
                use_bn: boolean, if True, apply BatchNormalization in each hidden layer
                n_cross_layers: int, num of cross layers
                seed: int, random value for initialization

            """
        super(DCNModel, self).__init__()
        self.cross_layer = CrossLayer(in_shape,
                                      n_layers=n_cross_layers,
                                      l2_reg=l2_reg)
        self.dnn_layer = DNNLayer(in_shape=in_shape,
                                  hidden_units=hidden_units,
                                  activation=act_fn,
                                  l2_reg=l2_reg,
                                  dropout_rate=dropout_rate,
                                  use_bn=use_bn,
                                  seed=seed)

    def forward(self, inputs):
        """
        Args:
            inputs: 2d tensor (batch_size, n_dim)

        Returns:
            2d tensor (batch_size, out_dim)

        """
        cross_output = self.cross_layer(inputs)
        dnn_output = self.dnn_layer(inputs)

        combined_output = torch.cat([cross_output, dnn_output], dim=-1)
        return combined_output
