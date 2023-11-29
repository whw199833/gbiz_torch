# coding: utf-8
# @Author: Haowen Wang

import torch
import torch.nn as nn
import torch.nn.functional as F
from gbiz_torch.layer import FuseLayer, DNNLayer, Dice


class DINModel(nn.Module):
    """
      Model: DIN Model

      Paper: Deep Interest Network for Click-Through Rate Prediction

      Link: https://arxiv.org/abs/1706.06978

      Author: Guorui Zhou, Chengru Song, Xiaoqiang Zhu, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, Kun Gai

      Developer: Haowen Wang

      inputs: list with length of 2, [input_a, input_b]
          input_a and input_b must have the same shape, (batch, a_len, dim)

      outputs: (batch, a_len, output_size)

      """

    def __init__(self,
                 in_shape,
                 projection_in_shape,
                 input_length,
                 hidden_units=[16, 4],
                 projection_hidden_units=[4, 1],
                 apply_final_act=False,
                 act_fn='relu',
                 dropout_rate=0,
                 l2_reg=0.001,
                 use_bn=False,
                 weight_norm=False,
                 seed=1024):
        """
            Args:
                input_length: int, sequence length of behaviour features
                hidden_units: list, units for the MLP layer
                act_fn: string, activation function
                l2_reg: float, regularization value
                dropout_rate: float, dropout rate
                use_bn: boolean, if True use BatchNormalization, otherwise not
                seed: int, random value
                weight_norm: boolean, if True scale the attention score
                projection_hidden_units: list, units for the projection layer
                apply_final_act: whether to apply act in final layer
            """
        super(DINModel, self).__init__()
        self.input_length = input_length
        self.din_layer = FuseLayer(in_shape,
                                   input_length=input_length,
                                   hidden_units=hidden_units,
                                   act_fn=act_fn,
                                   l2_reg=l2_reg,
                                   dropout_rate=dropout_rate,
                                   use_bn=use_bn,
                                   seed=seed,
                                   weight_norm=weight_norm)

        projection_hidden_units = [
            projection_in_shape] + projection_hidden_units
        self.dnn_layer = nn.ModuleList([
            nn.Linear(projection_hidden_units[i],
                      projection_hidden_units[i + 1])
            for i in range(len(projection_hidden_units) - 1)
        ])
        self.act_fn_layer = nn.ModuleList([
            Dice(in_shape=projection_hidden_units[i + 1],
                 input_length=projection_hidden_units[i + 1])
            for i in range(len(projection_hidden_units) - 1)
        ])

    def forward(self, inputs, extra_input=None):
        """
        Args:
            inputs: list of 3d tensor (batch, seq_len, hidden_units)
            extra_input: 2d tensor, (batch, dim), user, item and context features
        Returns:
            2d tensor (batch_size, out_dim)

        """

        din_output = self.din_layer(inputs)

        if din_output.dim() > 2:
            din_output = torch.flatten(din_output, start_dim=1)
        if extra_input is not None:
            combined_input = torch.cat([din_output, extra_input], dim=-1)
        else:
            combined_input = din_output

        output = combined_input
        # print(f"combined_input shape is {combined_input.shape}")
        for i in range(len(self.dnn_layer)):
            output = self.dnn_layer[i](output)
            output = self.act_fn_layer[i](output)

        return output
