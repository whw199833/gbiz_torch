# coding: utf-8
# @Author: Haowen Wang

import torch
import torch.nn as nn
import torch.nn.functional as F
from gbiz_torch.layer import DNNLayer, BridgeLayer


class EDCNModel(nn.Module):
    """
      Model: EDCN Model

      Paper: Enhancing Explicit and Implicit Feature Interactions via Information Sharing for Parallel Deep CTR Models

      Link: https://dl.acm.org/doi/abs/10.1145/3459637.3481915

      Author: Chen, Bo and Wang, Yichao and Liu, Zhirong and Tang, Ruiming and Guo, Wei and Zheng, Hongkun and Yao, Weiwei and Zhang, Muyu and He, Xiuqiang

      Developer: Haowen Wang

      inputs: 2d tensor (batch_size, n_dim)

      outputs: 2d tensor (batch_size, out_dim)

    """

    def __init__(self,
                 in_shape,
                 projection_hidden_units=[4, 1],
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
        super(EDCNModel, self).__init__()
        self.bridge_layer = BridgeLayer(in_shape,
                                        n_layers=n_cross_layers,
                                        l2_reg=l2_reg)
        self.dnn_layer = DNNLayer(in_shape,
                                  hidden_units=projection_hidden_units,
                                  activation=act_fn,
                                  l2_reg=l2_reg,
                                  dropout_rate=dropout_rate,
                                  use_bn=use_bn,
                                  seed=seed,
                                  apply_final_act=False)

    def forward(self, inputs):
        """
            Args:
                inputs: 2d tensor (batch_size, n_dim)

            Returns:
                2d tensor (batch_size, out_dim)

            """
        cross_output = self.bridge_layer(inputs)
        # print('cross_output ', cross_output.shape)
        output = self.dnn_layer(cross_output)
        return output
