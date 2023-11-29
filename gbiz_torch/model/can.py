# coding: utf-8
# @Author: Haowen Wang

import torch
import torch.nn as nn
import torch.nn.functional as F
from gbiz_torch.layer import DNNLayer, CoActionLayer


class CANModel(nn.Module):
    """
      MModel: CAN model

      Paper: CAN: Revisiting Feature Co-Action for Click-Through Rate Prediction

      Link: https://arxiv.org/abs/2011.05625

      Author: Guorui Zhou, Weijie Bian, et.al

      Developer: Haowen Wang

      inputs:
          2d tensor (batch_size, dim_1)

      outputs:
          2d tensor (batch_size, out_dim)

      """

    def __init__(self,
                 in_shape_list,
                 projection_in_shape,
                 hidden_units=[8],
                 use_bias=True,
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
        super(CANModel, self).__init__()
        self.co_action_layer = CoActionLayer(in_shape_list=in_shape_list,
                                             hidden_units=hidden_units,
                                             apply_final_act=True,
                                             use_bias=use_bias)
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
            inputs: tuple, (input_a, input_b)
            input_a: (batch, seq, dim1) or (batch, dim1)
            input_b: (batch, dim2)

            dim2 >> dim1

        Returns:
            2d tensor (batch_size, out_dim)

        """
        co_action_output = self.co_action_layer(inputs)
        # print(f"co_action_output.shape is {co_action_output.shape}")

        if extra_input is not None:
            if extra_input.dim() < 3:
                extra_input = torch.tile(torch.unsqueeze(
                    extra_input, dim=1), dims=(1, co_action_output.shape[1], 1))
            # if extra_input.dim() > 2:
            #     extra_input = torch.flatten(extra_input, start_dim=1)
            combined_input = torch.cat([co_action_output, extra_input], dim=-1)

        else:
            combined_input = co_action_output
        # print(f"combined_input.shape is {combined_input.shape}")

        output = self.dnn_layer(combined_input)
        return output
