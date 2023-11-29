# coding: utf-8
# @Author: Haowen Wang

import torch
import torch.nn as nn
import torch.nn.functional as F
from gbiz_torch.layer import DNNLayer


class GRU4RecModel(nn.Module):
    """
      Model: GRU4Rec Model

      Paper: Session-based Recommendations with Recurrent Neural Networks

      Link: https://arxiv.org/abs/1511.06939

      Author: Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, Domonkos Tikk

      Developer: anbo

      Date: 2022-02-08

      inputs: (batch, seq_len, hidden_units)

      output: (batch, hidden_units)

      """

    def __init__(self,
                 in_shape,
                 projection_in_shape,
                 rnn_unit,
                 projection_hidden_units=[8, 1],
                 rnn_act_fn='tanh',
                 act_fn='relu',
                 return_sequences=False,
                 l2_reg=0.001,
                 dropout_rate=0,
                 apply_final_act=False,
                 use_bn=False,
                 seed=1024):
        """
            Args:
                rnn_unit: int, rnn hidden units
                rnn_act_fn: string, activation function in rnn
                return_sequences: bool, whether return hidden states of rnn
                apply_final_act: whether to apply act in final layer
                projection_hidden_units: list, unit in each hidden layer
                act_fn: string, activation function
                l2_reg: float, regularization value
                dropout_rate: float, fraction of the units to dropout.
                use_bn: boolean, if True, apply BatchNormalization in each hidden layer
                seed: int, random value for initialization

            """
        super(GRU4RecModel, self).__init__()
        self.rnn_layer = nn.GRU(in_shape, rnn_unit, 1)

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
              inputs: 3d tensor (batch_size, len, dim_1), sequence features
              extra_input: 2d tensor (batch_size, dim_2), wide features

          Returns:
              2d tensor (batch_size, out_dim)

        """
        rnn_output = self.rnn_layer(inputs)
        rnn_output = F.tanh(rnn_output[0])
        # print('rnn_output ', rnn_output.shape)

        if rnn_output.dim() > 2:
            rnn_output = torch.flatten(rnn_output, start_dim=1)
        # print('rnn_output ', rnn_output.shape)

        if extra_input is not None:
            rnn_output = torch.cat([rnn_output, extra_input], dim=-1)

        output = self.dnn_layer(rnn_output)
        return output
