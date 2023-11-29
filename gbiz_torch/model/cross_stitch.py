# coding: utf-8
# @Author: Haowen Wang

import torch
import torch.nn as nn
import torch.nn.functional as F
from gbiz_torch.layer import DNNLayer, CrossStitchLayer


class CrossStitchModel(nn.Module):
    """
      Model: Cross Stitch Model

      Paper: Cross-stitch Networks for Multi-task Learning

      Link: https://arxiv.org/pdf/1604.03539

      Author: Ishan Misra, Abhinav Shrivastava, Abhinav Gupta, Martial Hebert

      Developer: Haowen Wang

      inputs:
          2d tensor (batch_size, dim_1)

      outputs:
          list of 2d tensor (batch_size, out_dim)

      """

    def __init__(
        self,
        in_shape,
        sharing_hidden_units=[16],
        task_hidden_units=[[1], [1]],
        task_apply_final_act=False,
        act_fn='relu',
        dropout_rate=0,
        use_bn=False,
        l2_reg=0.001,
        seed=1024,
    ):
        """
        Args:
            sharing_hidden_units: list, unit in each hidden layer
            task_hidden_units: list of list, unit of each hidden layer in task specific layers
            cross_stitch_layers_list: list of int, maximum should be <= min(layers of task_hidden_units)
            act_fn: string, activation function
            l2_reg: float, regularization value
            dropout_rate: float, fraction of the units to dropout.
            use_bn: boolean, if True, apply BatchNormalization in each hidden layer
            task_apply_final_act: bool
            seed: int, random value for initialization

        """
        super(CrossStitchModel, self).__init__()
        self.sharing_layer = DNNLayer(in_shape=in_shape,
                                      hidden_units=sharing_hidden_units,
                                      activation=act_fn,
                                      l2_reg=l2_reg,
                                      dropout_rate=dropout_rate,
                                      use_bn=use_bn,
                                      seed=seed)

        self.n_tasks = len(task_hidden_units)

        self.n_cross_stitch_layers = min(
            [len(units) for units in task_hidden_units])

        cross_in_shape_list = []

        for i in range(self.n_cross_stitch_layers):
            temp_list = []
            for j in range(self.n_tasks):
                temp_list.append(task_hidden_units[j][i])
                # cross_in_shape_list[i].append(task_hidden_units[j][i])
            cross_in_shape_list.append(temp_list)
        # print('cross_in_shape_list ', cross_in_shape_list)

        self.cross_stitch_layers = nn.ModuleList([
            CrossStitchLayer(in_shape_list=units)
            for i, units in enumerate(cross_in_shape_list)
        ])

        self.task_layers = nn.ModuleList()

        for i, units in enumerate(task_hidden_units):
            self.task_layers.append(nn.ModuleList())
            input_dim = sharing_hidden_units[-1]
            for j, unit in enumerate(units):
                output_dim = unit
                self.task_layers[i].append(
                    nn.Linear(input_dim, output_dim, bias=True))
                input_dim = output_dim

    def forward(self, inputs):
        """
        Args:
            inputs: 2d tensor (batch_size, dim_1), deep features

        Returns:
            list of 2d tensor (batch_size, out_dim)

        """
        sharing_output = self.sharing_layer(inputs)
        print('sharing_output ', sharing_output.shape)

        task_outputs = [sharing_output] * self.n_tasks
        for i in range(self.n_cross_stitch_layers):
            for j in range(self.n_tasks):
                task_outputs[j] = self.task_layers[j][i](task_outputs[j])
                if i < self.n_cross_stitch_layers - 1:
                    task_outputs[j] = F.relu(task_outputs[j])
                print('task_outputs ', task_outputs[j].shape)

            task_outputs = self.cross_stitch_layers[i](task_outputs)
        return task_outputs
