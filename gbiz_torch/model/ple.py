# coding: utf-8
# @Author: Haowen Wang

import torch
import torch.nn as nn
import torch.nn.functional as F
from gbiz_torch.layer import DNNLayer, CGCGatingNetworkLayer, ParallelDNNLayer


class PLEModel(nn.Module):
    """
      Model: PLE model

      Paper: Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations

      Link: dl.acm.org/doi/10.1145/3383313.3412236

      Author: Hongyan Tang, Junning Liu, Ming Zhao, Xudong Gong

      Developer: Haowen Wang

      inputs:
          2d tensor (batch_size, input_dim)

      outputs:
          list of 2d tensor (batch_size, out_dim)

      """

    def __init__(self,
                 in_shape,
                 expert_units,
                 n_task_experts,
                 n_shared_experts,
                 task_hidden_units,
                 task_apply_final_act=False,
                 act_fn='relu',
                 l2_reg=0.001,
                 dropout_rate=0,
                 use_bn=False,
                 seed=1024):
        """
            Args:
                expert_units: list of int, unit of each hidden layer in expert layers
                n_task_experts: list of list of int, n_task_experts in each task-specific Expert domain
                n_shared_experts: list of int, n_shared_experts in each shared expert domain
                task_hidden_units: list of list, unit of each hidden layer in task specific layers
                act_fn: string, activation function
                l2_reg: float, regularization value
                dropout_rate: float, fraction of the units to dropout.
                use_bn: boolean, if True, apply BatchNormalization in each hidden layer
                task_apply_final_act: bool
                seed: int, random value for initialization

            """
        super(PLEModel, self).__init__()
        self.num_tasks = len(task_hidden_units)
        # there is self.num_tasks final tasks and expert domains, plus one shared expert domain
        self.num_cgc_layers = len(expert_units)
        # need special treatement in the first and final layers
        # there is only one input in the first cgc layer
        # there are multiple inputs from each expert domain in the middle layers
        # there is no need to calculate gating network in shared expert in the last layer
        assert ((len(n_task_experts) == len(n_shared_experts))
                and (len(n_task_experts) == self.num_cgc_layers)), (
                    'There must be the same number of cgc_layers')

        for i, units in enumerate(n_task_experts):
            assert (len(units) == self.num_tasks), (
                'There must be equal number of task-specific expert domain as num of tasks'
            )

        self.cgc_gating_layers = nn.ModuleList()
        self.cgc_shared_expert_layers, self.cgc_task_expert_layers = nn.ModuleList(
        ), nn.ModuleList()
        for i in range(self.num_cgc_layers):
            input_dim = in_shape if i == 0 else expert_units[i - 1]
            self.cgc_task_expert_layers.append(
                nn.ModuleList([
                    ParallelDNNLayer(in_shape=input_dim,
                                     hidden_units=[expert_units[i]],
                                     activation=act_fn,
                                     l2_reg=l2_reg,
                                     dropout_rate=dropout_rate,
                                     use_bn=use_bn,
                                     apply_final_act=True,
                                     seed=seed,
                                     n_experts=unit)
                    for j, unit in enumerate(n_task_experts[i])
                ]))

            self.cgc_shared_expert_layers.append(
                ParallelDNNLayer(in_shape=input_dim,
                                 hidden_units=[expert_units[i]],
                                 activation=act_fn,
                                 l2_reg=l2_reg,
                                 dropout_rate=dropout_rate,
                                 use_bn=use_bn,
                                 apply_final_act=True,
                                 seed=seed,
                                 n_experts=n_shared_experts[i]))

            n_cgc_gatings = self.num_tasks + 1 if i < self.num_cgc_layers - 1 else self.num_tasks
            # by default, last cgc gating network in each cgc layer is shared expert domain

            total_experts = [1] * n_cgc_gatings
            for j in range(n_cgc_gatings):
                if j < self.num_tasks:
                    total_experts[j] = n_task_experts[i][j] + \
                        n_shared_experts[i]
                else:
                    total_experts[j] = sum(
                        n_task_experts[i]) + n_shared_experts[i]

            self.cgc_gating_layers.append(
                nn.ModuleList([
                    CGCGatingNetworkLayer(input_dim,
                                          total_experts=total_experts[j],
                                          l2_reg=l2_reg,
                                          seed=seed) for j in range(n_cgc_gatings)
                ]))

        self.task_layers = nn.ModuleList([
            DNNLayer(in_shape=expert_units[-1],
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
        task_expert_outputs = []
        shared_expert_output = inputs
        for i in range(self.num_cgc_layers):
            shared_expert_input = shared_expert_output
            shared_expert_output = self.cgc_shared_expert_layers[i](
                shared_expert_input)
            # shared_expert_output: (batch, n_shared_experts, dim)

            tmp_task_expert_outputs = []
            for j in range(self.num_tasks):
                # calculate task-specific experts
                if i == 0:
                    task_expert_input = inputs
                    task_expert_output = self.cgc_task_expert_layers[i][j](
                        task_expert_input)
                    # task_expert_outputs.append(task_expert_output)
                else:
                    task_expert_input = task_expert_outputs[j]
                    task_expert_output = self.cgc_task_expert_layers[i][j](
                        task_expert_input)
                    # task_expert_outputs[j] = task_expert_output
                # task_expert_output: (batch, n_task_experts, dim)
                tmp_task_expert_outputs.append(task_expert_output)

                # calculate task-shared gating net output
                task_cgc_gating_network_output = self.cgc_gating_layers[i][j](
                    [task_expert_output, shared_expert_output, task_expert_input])
                # task_cgc_gating_network_output: (batch, dim)

                if i == 0:
                    task_expert_outputs.append(task_cgc_gating_network_output)
                else:
                    task_expert_outputs[j] = task_cgc_gating_network_output

            if i < self.num_cgc_layers - 1:
                combined_task_expert_output = torch.cat(
                    tmp_task_expert_outputs, dim=1)
                shared_expert_output = self.cgc_gating_layers[i][-1]([
                    combined_task_expert_output, shared_expert_output,
                    shared_expert_input
                ])
                # print('shared_expert_output {}'.format(shared_expert_output.shape))

        task_outputs = []
        for i in range(self.num_tasks):
            task_outputs.append(self.task_layers[i](task_expert_outputs[i]))

        return task_outputs
