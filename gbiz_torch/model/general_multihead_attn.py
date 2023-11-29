# coding: utf-8
# @Author: Haowen Wang

import torch
import torch.nn as nn
import torch.nn.functional as F
from gbiz_torch.layer import MultiHeadAttentionLayer, DNNLayer


class GeneralMultiHeadAttnModel(nn.Module):
    """
      Model: General Multi Head Attention Model

      Developer: Haowen Wang

      inputs: (batch, seq_len, hidden_units)

      output: (batch, hidden_units)

      """

    def __init__(self,
                 in_shape,
                 projection_in_shape,
                 hidden_units=16,
                 heads=4,
                 l2_reg=0.001,
                 dropout_rate=0,
                 act_fn='relu',
                 list_tag=True,
                 projection_hidden_units=[4, 1],
                 apply_final_act=False,
                 use_bn=False,
                 seed=1024,
                 mha_type='origin',
                 dim_e=3,
                 synthesizer_type='dense',
                 inf=1e9,
                 operation_type='concat'):
        """
            Args:
                hidden_units: int, the last dim of the inputs
                heads: int, num of self-attention modules, must be dividable by hidden_units
                act_fn: string, activation function
                dropout_rate: float, dropout rate
                l2_reg: float, regularization value
                projection_hidden_units: list, units for the projection layer
                apply_final_act: whether to apply act in final layer
                use_bn: boolean, if True use BatchNormalization, otherwise not
                seed: int, random value
            """
        super(GeneralMultiHeadAttnModel, self).__init__()
        self.operation_type = operation_type

        self.bert_layer = MultiHeadAttentionLayer(
            in_shape,
            hidden_units=hidden_units,
            heads=heads,
            l2_reg=l2_reg,
            dropout_rate=dropout_rate,
            mha_type=mha_type,
            dim_e=dim_e,
            synthesizer_type=synthesizer_type,
            inf=inf)

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
            inputs: 3d tensor (batch, seq_len, hidden_units)

        Returns:
            2d tensor (batch_size, out_dim)

        """
        bert_output = self.bert_layer(inputs)

        if bert_output.dim() > 2:
            if self.operation_type == 'concat':
                bert_output = torch.flatten(bert_output, start_dim=1)
            else:
                bert_output = torch.sum(bert_output, dim=1, keepdim=False)

        if extra_input is not None:
            bert_output = torch.cat([bert_output, extra_input], dim=-1)

        output = self.dnn_layer(bert_output)
        return output
