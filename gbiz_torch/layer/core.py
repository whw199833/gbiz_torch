import torch
import torch.nn as nn
import torch.nn.functional as F


class DNNLayer(nn.Module):
    """
      Model: The Multi Layer Perceptron

      Developer: Haowen Wang

      Input shape
          - nD tensor with shape: ``(batch_size, ..., input_dim)``.
          - The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.

      Output shape
          - nD tensor with shape: ``(batch_size, ..., hidden_units[-1])``.
          - For instance, for a 2D input with shape ``(batch_size, input_dim)``,
              the output would have shape ``(batch_size, hidden_units[-1])``.
    """

    def __init__(self,
                 in_shape,
                 hidden_units=[16, 1],
                 activation='relu',
                 init_std=0.0001,
                 l2_reg=0,
                 dropout_rate=0,
                 use_bn=False,
                 apply_final_act=True,
                 seed=1024):
        """
            Args:
                hidden_units: list of positive integer, the layer number and units in each layer.
                activation: Activation function to use.
                l2_reg: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix.
                dropout_rate: float in [0,1). Fraction of the units to dropout.
                use_bn: bool. Whether use BatchNormalization before activation or not.
                apply_final_act: whether to apply act in final layer
                seed: A Python integer to use as random seed.

        """
        super(DNNLayer, self).__init__()
        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        self.apply_final_act = apply_final_act
        self.in_shape = in_shape
        self.init_std = init_std
        self.build()
        self.reset_parameters()

    def build(self):
        hidden_units = [int(self.in_shape)] + list(self.hidden_units)

        self.linear_layers = nn.ModuleList([
            nn.Linear(hidden_units[i], hidden_units[i + 1])
            for i in range(len(self.hidden_units))
        ])

        if self.activation == 'relu':
            self.act_layers = nn.ModuleList(
                [nn.ReLU() for i in range(len(self.hidden_units))]
            )
        elif self.activation == 'leaky_relu':
            self.act_layers = nn.ModuleList(
                [nn.LeakyReLU() for i in range(len(self.hidden_units))]
            )

        if self.use_bn:
            self.bn_layers = nn.ModuleList([
                nn.BatchNorm1d(self.hidden_units[i])
                for i in range(len(self.hidden_units))
            ])

        if self.dropout_rate is not None and self.dropout_rate > 0:
            self.dropout_layers = nn.ModuleList([
                nn.Dropout(self.dropout_rate) for i in range(len(self.hidden_units))
            ])

    def reset_parameters(self):
        for name, val in self.linear_layers.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(val)

    def forward(self, inputs):
        deep_input = inputs

        for i in range(len(self.hidden_units)):
            fc = self.linear_layers[i](deep_input)

            if self.use_bn:
                fc = self.bn_layers[i](fc)

            if i < len(self.hidden_units) - 1 or self.apply_final_act:
                fc = self.act_layers[i](fc)
                # fc = F.leaky_relu(fc)

            if self.dropout_rate is not None and self.dropout_rate > 0:
                fc = self.dropout_layers[i](fc)

            deep_input = fc

        return deep_input


class GeneralMMoELayer(nn.Module):
    """
      Model: General MMoE Layer

      Paper: Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts

      Link: www.kdd.org/kdd2018/accepted-papers/view/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-

      Author: Jiaqi Ma, Zhe Zhao, Xinyang Yi, Jilin Chen, Lichan Hong, Ed Chi

      Developer: Haowen Wang

      inputs: list of tensors, [shared_outputs, experts_output]
              shared_outputs: (batch, dim_1)
              experts_output: (batch, dim_2, n_experts)
      returns:
          list of tensors (N, ..., dim_i)

    """

    def __init__(self,
                 shared_in_shape,
                 expert_in_shape,
                 num_experts,
                 num_tasks,
                 bias_init=[],
                 l2_reg=0.001,
                 seed=1024):
        super(GeneralMMoELayer, self).__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.l2_reg = l2_reg
        self.seed = seed
        self.bias_init = bias_init
        self.shared_in_shape = shared_in_shape
        self.expert_in_shape = expert_in_shape
        self.build()
        self.reset_parameters()

    def build(self):
        self.gate_kernels = nn.Parameter(
            torch.empty((self.shared_in_shape, self.num_experts, self.num_tasks)))
        self.gate_bias = nn.Parameter(
            torch.empty((self.num_experts, self.num_tasks)))

    def reset_parameters(self):
        nn.init.xavier_normal_(self.gate_kernels)
        nn.init.zeros_(self.gate_bias)

    def forward(self, inputs):
        """
        Args:
            inputs: list of tensors, [shared_outputs, experts_output]
                shared_outputs: (batch, dim_1)
                experts_output: (batch, dim_2, n_experts)
        returns:
            list of tensors (N, ..., dim_i)
        """
        shared_outputs, expert_outputs = inputs
        input_ndims = expert_outputs.dim()

        # gate_output = torch.matmul(shared_outputs, self.gate_kernels)
        gate_output = torch.einsum('ac,cde->ade',
                                   (shared_outputs, self.gate_kernels))
        # gate_output: (bz, num_experts, num_tasks)
        print('gate_output ', gate_output.shape)
        gate_output = torch.add(gate_output, self.gate_bias)
        gate_score = F.softmax(gate_output, dim=1)
        # gate_score: (bz, num_experts, num_tasks)
        # expert_outputs: (bz, dim_2, num_experts)
        final_outputs = torch.einsum(
            'abd,ade->abe', (expert_outputs, gate_score))
        # final_outputs: (bz, dim_2, num_tasks)

        if input_ndims == 4:
            return [final_outputs[:, :, :, i] for i in range(self.num_tasks)]
        elif input_ndims == 3:
            return [final_outputs[:, :, i] for i in range(self.num_tasks)]
