# coding: utf-8
# @Author: Haowen Wang

import torch
import torch.nn as nn
import torch.nn.functional as F
from gbiz_torch.layer import DNNLayer


class WndModel(nn.Module):
    """
      Model: Wnd Model

      Paper: Wide & Deep Learning for Recommender Systems

      Link: https://arxiv.org/abs/1606.07792

      Author: Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, etc.

      Developer: Haowen Wang

      inputs:
          2d tensor (batch_size, dim_1)

      outputs:
          2d tensor (batch_size, out_dim)

      """

    def __init__(self,
                 in_shape,
                 hidden_units,
                 act_fn='relu',
                 l2_reg=0.001,
                 dropout_rate=0,
                 use_bn=False,
                 seed=1024,
                 apply_final_act=True):
        """
            Args:
                hidden_units: list, unit in each hidden layer
                act_fn: string, activation function
                l2_reg: float, regularization value
                dropout_rate: float, fraction of the units to dropout.
                use_bn: boolean, if True, apply BatchNormalization in each hidden layer
                seed: int, random value for initialization

            """
        super(WndModel, self).__init__()
        self.dnn_layer = DNNLayer(in_shape,
                                  hidden_units=hidden_units,
                                  activation=act_fn,
                                  l2_reg=l2_reg,
                                  dropout_rate=dropout_rate,
                                  use_bn=use_bn,
                                  seed=seed,
                                  apply_final_act=apply_final_act)

    def forward(self, inputs, extra_input=None):
        """
            Args:
                inputs: 2d tensor (batch_size, dim_1), deep features
                extra_input: 2d tensor (batch_size, dim_2), wide features

            Returns:
                2d tensor (batch_size, out_dim)

            """
        dnn_output = self.dnn_layer(inputs)

        if extra_input is None:
            combined_output = dnn_output
        else:
            combined_output = torch.cat([dnn_output, extra_input], dim=-1)
        return combined_output
