# coding: utf-8
# @Author: Haowen Wang

import torch
import torch.nn as nn
import torch.nn.functional as F
from gbiz_torch.layer import DNNLayer, ContextNetBlockLayer


class ContextNetModel(nn.Module):
    """
      Model: ContextNet model

      Paper: ContextNet: A Click-Through Rate Prediction Framework Using Contextual information to Refine Feature Embedding

      Link: https://arxiv.org/abs/2107.12025

      Author: Zhiqiang Wang, Qingyun She, PengTao Zhang, Junlin Zhang

      Developer: Haowen Wang

      inputs: inputs: (batch, fields, dim1)

      outputs:
          2d tensor (batch_size, out_dim)

      """

    def __init__(self,
                 fields,
                 in_shape,
                 projection_in_shape,
                 hidden_size=32,
                 share_aggregation_tag=True,
                 point_fnn_tag=False,
                 contextnet_block_layers=1,
                 projection_hidden_units=[4, 1],
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
        super(ContextNetModel, self).__init__()

        self.contextnet_block_layer = ContextNetBlockLayer(
            fields,
            in_shape,
            hidden_size=hidden_size,
            share_aggregation_tag=share_aggregation_tag,
            point_fnn_tag=point_fnn_tag,
            contextnet_block_layers=contextnet_block_layers,
            l2_reg=l2_reg,
            seed=seed)

        self.dnn_layer = DNNLayer(in_shape=projection_in_shape,
                                  hidden_units=projection_hidden_units,
                                  activation=act_fn,
                                  l2_reg=l2_reg,
                                  dropout_rate=dropout_rate,
                                  use_bn=use_bn,
                                  apply_final_act=apply_final_act,
                                  seed=seed)

    def forward(self, inputs, extra_input=None):
        """
        Args:
            inputs: (batch, fields, dim1)

        Returns: 2d tensor (batch, out_dim)
        """
        contextnet_output = self.contextnet_block_layer(inputs)

        if extra_input is not None:
            if extra_input.dim() > 2:
                extra_input = torch.flatten(extra_input, start_dim=1)
            combined_input = torch.cat(
                [contextnet_output, extra_input], dim=-1)
        else:
            combined_input = contextnet_output

        output = self.dnn_layer(combined_input)
        return output
