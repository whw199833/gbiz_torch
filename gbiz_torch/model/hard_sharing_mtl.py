# coding: utf-8
# @Author: Haowen Wang

import torch
import torch.nn as nn
import torch.nn.functional as F
from gbiz_torch.layer import DNNLayer


class HardSharingModel(nn.Module):
    """
      Model: Hard Sharing Model

      Developer: Haowen Wang

      inputs:
          2d tensor (batch_size, dim_1)

      outputs:
          list of 2d tensor (batch_size, out_dim)

      """

    def __init__(self,
                 in_shape,
                 sharing_hidden_units,
                 task_hidden_units,
                 task_apply_final_act=False,
                 act_fn='relu',
                 l2_reg=0.001,
                 dropout_rate=0,
                 use_bn=False,
                 seed=1024):
        """
            Args:
                sharing_hidden_units: list, unit in each hidden layer
                task_hidden_units: list of list, unit of each hidden layer in task specific layers
                act_fn: string, activation function
                l2_reg: float, regularization value
                dropout_rate: float, fraction of the units to dropout.
                use_bn: boolean, if True, apply BatchNormalization in each hidden layer
                task_apply_final_act: bool
                seed: int, random value for initialization

            """
        super(HardSharingModel, self).__init__()
        self.sharing_layer = DNNLayer(in_shape,
                                      hidden_units=sharing_hidden_units,
                                      activation=act_fn,
                                      l2_reg=l2_reg,
                                      dropout_rate=dropout_rate,
                                      use_bn=use_bn,
                                      seed=seed)

        self.n_tasks = len(task_hidden_units)
        self.task_layers = nn.ModuleList([
            DNNLayer(in_shape=sharing_hidden_units[-1],
                     hidden_units=units,
                     activation=act_fn,
                     l2_reg=l2_reg,
                     dropout_rate=dropout_rate,
                     use_bn=use_bn,
                     apply_final_act=True,
                     seed=seed) for i, units in enumerate(task_hidden_units)
        ])

    def forward(self, inputs):
        """
            Args:
                inputs: 2d tensor (batch_size, dim_1), deep features

            Returns:
                list of 2d tensor (batch_size, out_dim)

            """
        sharing_output = self.sharing_layer(inputs)

        task_outputs = []
        for i in range(self.n_tasks):
            task_outputs.append(self.task_layers[i](sharing_output))

        return task_outputs
