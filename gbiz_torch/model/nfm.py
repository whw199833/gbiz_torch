# coding: utf-8
# @Author: Haowen Wang

import torch
import torch.nn as nn
import torch.nn.functional as F
from gbiz_torch.layer import DNNLayer, FMLayer


class NFMModel(nn.Module):
    """
      Model: NFM Model

      Paper: Neural Factorization Machines for Sparse Predictive Analytics

      Link: https://arxiv.org/abs/1708.05027

      Author: Xiangnan He, Tat-Seng Chua

      Developer: Haowen Wang

      inputs: 3d tensor (batch_size, fields, n_dim)

      outputs: 2d tensor (batch_size, out_dim)

      """

    def __init__(self,
                 in_shape,
                 hidden_units,
                 keep_fm_dim=False,
                 act_fn='relu',
                 l2_reg=0.001,
                 dropout_rate=0,
                 use_bn=False,
                 seed=1024,
                 apply_final_act=False):
        """
            Args:
                hidden_units: list, unit in each hidden layer
                act_fn: string, activation function
                l2_reg: float, regularization value
                dropout_rate: float, fraction of the units to dropout.
                use_bn: boolean, if True, apply BatchNormalization in each hidden layer
                seed: int, random value for initialization

            """
        super(NFMModel, self).__init__()
        self.fm_layer = FMLayer(keep_dim=keep_fm_dim)
        self.dnn_layer = DNNLayer(in_shape=1 if keep_fm_dim == False else in_shape,
                                  hidden_units=hidden_units,
                                  activation=act_fn,
                                  l2_reg=l2_reg,
                                  dropout_rate=dropout_rate,
                                  use_bn=use_bn,
                                  seed=seed,
                                  apply_final_act=apply_final_act)

    def forward(self, inputs):
        """
            Args:
                inputs: 3d tensor (batch_size, fields, n_dim)

            Returns:
                2d tensor (batch_size, out_dim)

            """
        fm_output = self.fm_layer(inputs)
        dnn_output = self.dnn_layer(fm_output)

        return dnn_output
