# coding: utf-8
# @Author: Haowen Wang

import torch
import torch.nn as nn
import torch.nn.functional as F
from gbiz_torch.layer import DNNLayer


class TowerModel(nn.Module):
    """
      Model: Tower Model

      Developer: Haowen Wang

      inputs: list of 2d tensor [input_a, input_b]
               input_a: (batch_size, dim_1)
               input_b: (batch_size, dim_2)

      outputs: 2d tensor (batch_size, 1)

      """

    def __init__(self,
                 in_shape_list,
                 user_hidden_units,
                 item_hidden_units,
                 act_fn='relu',
                 l2_reg=0.001,
                 dropout_rate=0,
                 use_bn=False,
                 seed=1024):
        """
            Args:
                user_hidden_units: list, unit in user tower hidden layer
                item_hidden_units: list, unit in item tower hidden layer
                act_fn: string, activation function
                l2_reg: float, regularization value
                dropout_rate: float, fraction of the units to dropout.
                use_bn: boolean, if True, apply BatchNormalization in each hidden layer
                seed: int, random value for initialization

            """
        super(TowerModel, self).__init__()
        assert (user_hidden_units is not None and len(user_hidden_units) >= 1), (
            'Must specify user_hidden_units with at least one unit')
        assert (item_hidden_units is not None and len(item_hidden_units) >= 1), (
            'Must specify item_hidden_units with at least one unit')
        assert (user_hidden_units[-1] == item_hidden_units[-1]), (
            'the last unit of user_hidden_units must equal to that of item_hidden_units'
        )

        self.user_hidden_units = user_hidden_units
        self.item_hidden_units = item_hidden_units
        self.user_tower = DNNLayer(
            in_shape=in_shape_list[0],
            hidden_units=self.user_hidden_units,
            activation=act_fn,
            l2_reg=l2_reg,
            dropout_rate=dropout_rate,
            use_bn=use_bn,
            seed=seed) if len(self.user_hidden_units) > 0 else None

        self.item_tower = DNNLayer(
            in_shape=in_shape_list[1],
            hidden_units=self.item_hidden_units,
            activation=act_fn,
            l2_reg=l2_reg,
            dropout_rate=dropout_rate,
            use_bn=use_bn,
            seed=seed) if len(self.item_hidden_units) > 0 else None

        user_last_dim = user_hidden_units.pop()
        item_last_dim = item_hidden_units.pop()
        self.user_last_layer = nn.Linear(user_hidden_units[-1], user_last_dim)
        self.item_last_layer = nn.Linear(item_hidden_units[-1], item_last_dim)

    def forward(self, inputs):
        """
            Args:
                inputs: list of 2d tensor [input_a, input_b]
                 input_a: (batch_size, dim_1)
                 input_b: (batch_size, dim_2)

            Returns:
                2d tensor (batch_size, 1)

            """
        user_inputs, item_inputs = inputs

        user_hidden = self.user_tower(
            user_inputs) if self.user_tower is not None else user_inputs
        item_hidden = self.item_tower(
            item_inputs) if self.item_tower is not None else item_inputs

        user_output = self.user_last_layer(user_hidden)
        item_output = self.item_last_layer(item_hidden)

        logit = torch.sum(torch.mul(user_output, item_output), dim=-1)
        logit = torch.reshape(logit, (-1, 1))

        return logit
