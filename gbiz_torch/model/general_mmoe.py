# coding: utf-8
# @Author: Haowen Wang

import torch
import torch.nn as nn
import torch.nn.functional as F
from gbiz_torch.layer import DNNLayer, GeneralMMoELayer


class GeneralMMoEModel(nn.Module):
    """
      Model: General MMoE Model

      Paper: Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts

      Link: dl.acm.org/doi/10.1145/3219819.3220007

      Author: Jiaqi Ma, Zhe Zhao, Xinyang Yi, Jilin Chen, Lichan Hong, Ed Chi

      Developer: Haowen Wang

      inputs:
          2d tensor (batch_size, dim_1)

      outputs:
          list of 2d tensor (batch_size, out_dim)

      """

    def __init__(self,
                 in_shape,
                 experts_hidden_units,
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
                experts_hidden_units: list of list, unit of each hidden layer in expert layers
                sharing_hidden_units: list, unit in each hidden layer
                task_hidden_units: list of list, unit of each hidden layer in task specific layers
                act_fn: string, activation function
                l2_reg: float, regularization value
                dropout_rate: float, fraction of the units to dropout.
                use_bn: boolean, if True, apply BatchNormalization in each hidden layer
                task_apply_final_act: bool
                seed: int, random value for initialization

            """
        super(GeneralMMoEModel, self).__init__()
        self.num_experts = len(experts_hidden_units)
        self.num_tasks = len(task_hidden_units)
        tmp_flag = [ex[-1] for ex in experts_hidden_units]
        assert len(set(
            tmp_flag)) == 1, "the last units in each layer in experts_hidden_units should be the same"

        self.gmmoe_layer = GeneralMMoELayer(
            shared_in_shape=sharing_hidden_units[-1],
            expert_in_shape=experts_hidden_units[0][-1],
            num_experts=self.num_experts,
            num_tasks=self.num_tasks,
            l2_reg=l2_reg,
            seed=seed)

        self.sharing_layer = DNNLayer(in_shape,
                                      hidden_units=sharing_hidden_units,
                                      activation=act_fn,
                                      l2_reg=l2_reg,
                                      dropout_rate=dropout_rate,
                                      use_bn=use_bn,
                                      seed=seed)

        self.expert_layers = nn.ModuleList([
            DNNLayer(sharing_hidden_units[-1],
                     hidden_units=units,
                     activation=act_fn,
                     l2_reg=l2_reg,
                     dropout_rate=dropout_rate,
                     use_bn=use_bn,
                     apply_final_act=True,
                     seed=seed) for i, units in enumerate(experts_hidden_units)
        ])

        self.task_layers = nn.ModuleList([
            DNNLayer(in_shape=experts_hidden_units[0][-1],
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
        expert_inputs = self.sharing_layer(inputs)

        expert_outputs_list = []
        for i in range(self.num_experts):
            expert_outputs_i = self.expert_layers[i](expert_inputs)
            expert_outputs_i = torch.unsqueeze(expert_outputs_i, dim=-1)
            expert_outputs_list.append(expert_outputs_i)

        expert_outputs = torch.cat(expert_outputs_list, dim=-1)

        task_inputs = self.gmmoe_layer([expert_inputs, expert_outputs])

        task_outputs = []
        for i in range(self.num_tasks):
            task_outputs.append(self.task_layers[i](task_inputs[i]))

        return task_outputs
