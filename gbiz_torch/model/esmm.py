# coding: utf-8
# @Author: Haowen Wang

import torch
import torch.nn as nn
import torch.nn.functional as F
from gbiz_torch.layer import DNNLayer


class ESMMModel(nn.Module):
    """
      Model: ESMM Model

      Paper: Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate

      Link: https://arxiv.org/abs/1804.07931

      Author: Xiao Ma, Liqin Zhao, Guan Huang, Zhi Wang, Zelin Hu, Xiaoqiang Zhu, Kun Gai

      Developer: Haowen Wang

      inputs: 2d tensor (batch_size, n_dim)

      outputs: list of 2d tensor [input_a, input_b]
               input_a: (batch_size, 1)
               input_b: (batch_size, 1)

      """

    def __init__(self,
                 in_shape,
                 task_hidden_units=[[8, 1], [8, 1]],
                 task_apply_final_act=False,
                 act_fn='relu',
                 l2_reg=0.001,
                 dropout_rate=0,
                 use_bn=False,
                 seed=1024,
                 name='ESMMModel'):
        """
            Args:
                task_hidden_units: list of list, task_hidden_units[i] is list of hidden units in each task hidden layers
                task_apply_final_act, boolean, whether apply activation at the last layer of each task
                act_fn: string, activation function
                l2_reg: float, regularization value
                dropout_rate: float, fraction of the units to dropout.
                use_bn: boolean, if True, apply BatchNormalization in each hidden layer
                seed: int, random value for initialization

            """
        super(ESMMModel, self).__init__()
        assert (task_hidden_units is not None and len(task_hidden_units) > 1), (
            'Must specify task_hidden_units with at least one unit')

        self.num_tasks = len(task_hidden_units)

        self.task_layers = nn.ModuleList([
            DNNLayer(in_shape,
                     hidden_units=units,
                     activation=act_fn,
                     l2_reg=l2_reg,
                     dropout_rate=dropout_rate,
                     use_bn=use_bn,
                     apply_final_act=task_apply_final_act,
                     seed=seed) for i, units in enumerate(task_hidden_units)
        ])

    def forward(self, inputs):
        """
        Args:
            inputs: 2d tensor (batch_size, n_dim)

        Returns:
            list of 2d tensor [ctr_output, cvr_output]
                ctr_output: (batch_size, 1)
                cvr_output: (batch_size, 1)

        """
        task_outputs = []
        for i in range(self.num_tasks):
            task_outputs.append(self.task_layers[i](inputs))

        return task_outputs
