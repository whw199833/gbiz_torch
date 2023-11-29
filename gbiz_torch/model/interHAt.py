# coding: utf-8
# @Author: Haowen Wang

import torch
import torch.nn as nn
import torch.nn.functional as F
from gbiz_torch.layer import TransformerEncoder, DNNLayer, HierarchicalAttnAggLayer


class InterHAtModel(nn.Module):
    """
      Model: InterHAt Model

      Paper: Interpretable Click-Through Rate Prediction through Hierarchical Attention

      Link: https://dl.acm.org/doi/pdf/10.1145/3336191.3371785

      Author: Zeyu Li, Wei Cheng, Yang Chen, Haifeng Chen, Wei wang

      Developer: Haowen Wang

      inputs: (batch, fields, hidden_units)

      output: (batch, hidden_units)

      """

    def __init__(self,
                 in_shape,
                 projection_in_shape,
                 n_layers=4,
                 hidden_size=16,
                 hidden_units=16,
                 heads=4,
                 intermediate_size=64,
                 n_transform_layers=1,
                 act_fn='relu',
                 dropout_rate=0,
                 l2_reg=0.001,
                 return_all_layers=False,
                 projection_hidden_units=[4, 1],
                 apply_final_act=False,
                 use_bn=False,
                 seed=1024,
                 mha_type='origin'):
        """
            Args:
                n_transform_layers: int, num of transformer encoder layers
                hidden_units: int, the last dim of the inputs
                heads: int, num of self-attention modules, must be dividable by hidden_units
                intermediate_size: int, units in dense layer, normally intermediate_size=4*hidden_units
                act_fn: string, activation function
                dropout_rate: float, dropout rate
                l2_reg: float, regularization value
                return_all_layers: boolean, if True return list of (batch, seq_len, hidden_units), otherwise return (batch, seq_len, hidden_units)
                projection_hidden_units: list, units for the projection layer
                apply_final_act: whether to apply act in final layer
                use_bn: boolean, if True use BatchNormalization, otherwise not
                seed: int, random value

            """
        super(InterHAtModel, self).__init__()

        self.bert_layer = TransformerEncoder(in_shape,
                                             n_layers=n_transform_layers,
                                             hidden_units=hidden_units,
                                             heads=heads,
                                             intermediate_size=intermediate_size,
                                             act_fn=act_fn,
                                             dropout_rate=dropout_rate,
                                             l2_reg=l2_reg,
                                             return_all_layers=return_all_layers,
                                             mha_type=mha_type)

        self.hierarchicalattn_layer = HierarchicalAttnAggLayer(
            in_shape=in_shape,
            n_layers=n_layers,
            hidden_size=hidden_size,
            l2_reg=l2_reg,
            seed=seed,
        )

        self.dnn_layer = DNNLayer(
            in_shape=projection_in_shape,
            hidden_units=projection_hidden_units,
            activation=act_fn,
            l2_reg=l2_reg,
            dropout_rate=dropout_rate,
            use_bn=use_bn,
            apply_final_act=apply_final_act,
            seed=seed,
        )

    def forward(self, inputs, extra_input=None, lego='standard'):
        """
            Args:
                inputs: 3d tensor (batch, fields, hidden_units)

            Returns:
                2d tensor (batch_size, out_dim)

            """
        bert_output = self.bert_layer(inputs, lego=lego)

        hierarchical_attn_output = self.hierarchicalattn_layer(bert_output)

        if extra_input is not None:
            combined_input = torch.cat([hierarchical_attn_output, extra_input],
                                       dim=-1)

        else:
            combined_input = hierarchical_attn_output

        output = self.dnn_layer(combined_input)

        return output
