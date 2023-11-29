import torch
import torch.nn as nn
import torch.nn.functional as F


class CGCGatingNetworkLayer(nn.Module):
    """
      Model: gating network in Customized Gate Control (CGC)

      Paper: Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations

      Link: dl.acm.org/doi/10.1145/3383313.3412236

      Author: Hongyan Tang, Junning Liu, Ming Zhao, Xudong Gong

      Developer: Haowen Wang

      inputs:  task_expert_input: (batch, n_experts_1, dim_1)
           shared_expert_input: (batch, n_experts_2, dim_1)
           input: (batch, dim)

      output: (batch, dim_1)

    """

    def __init__(self, in_shape, total_experts, l2_reg=0, seed=1024):
        super(CGCGatingNetworkLayer, self).__init__()
        self.l2_reg = l2_reg
        self.seed = seed
        self.total_experts = total_experts
        self.in_shape = in_shape
        self.build()
        self.reset_parameters()

    def build(self):
        self.weight_w = nn.Parameter(
            torch.empty((self.in_shape, self.total_experts)))

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight_w)

    def forward(self, inputs):
        """
        :param inputs: list of 3 tensors
        :param training:
        :param kwargs:
        :return: (batch, dim_1)
        """
        task_expert_input, shared_expert_input, input = inputs
        # print('task_expert_input {}, shared_expert_input {}, input {}'.format(
        #     task_expert_input.shape, shared_expert_input.shape, input.shape))
        # task_expert_input: (batch, n_experts_1, dim_1)
        # shared_expert_input: (batch, n_experts_2, dim_1)
        # input: (batch, dim)

        combined_input = torch.cat(
            [task_expert_input, shared_expert_input], dim=1)
        # combined_input: (batch, total_experts, dim_1)
        # combined_input = torch.permute(combined_input, (0, 2, 1))
        combined_input = combined_input.permute((0, 2, 1))

        # print('self.weight_w ', self.weight_w.shape)
        gate_output = torch.matmul(input, self.weight_w)
        # print('gate_output ', gate_output.shape)
        gate_act = F.softmax(gate_output, dim=1)
        # print('gate_act ', gate_act.shape)
        gate_act = torch.unsqueeze(gate_act, dim=2)

        # print('combined_input {}, gate_act {}'.format(combined_input.shape,
        #                                               gate_act.shape))
        # final_outputs = torch.matmul(combined_input, gate_act)
        final_outputs = torch.einsum(
            'abc,ace->abe', (combined_input, gate_act))
        final_outputs = torch.squeeze(final_outputs, dim=-1)

        return final_outputs


class BiLinearInteractionLayer(nn.Module):
    """
      Model: BiLinear Interaction Layer in FiBiNet

      Paper: FiBiNET- Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction

      Link: https://arxiv.org/abs/1905.09433

      Author: Tongwen Huang, Zhiqi Zhang, Junlin Zhang

      Developer: Haowen Wang

      attn = V

      inputs: (batch, seq_len, dim)

      return: (batch, seq_len * (seq_len-1) // 2, dim)
      """

    def __init__(self,
                 in_shape,
                 bilinear_type='field_all',
                 l2_reg=0.001,
                 seed=1024):
        self.l2_reg = l2_reg
        self.seed = seed
        self.bilinear_type = bilinear_type
        self.in_shape = in_shape
        super(BiLinearInteractionLayer, self).__init__()
        self.build()
        self.reset_parameters()

    def build(self):
        if self.bilinear_type == 'field_all':
            self.weight = nn.Parameter(
                torch.empty((self.in_shape, self.in_shape)))
        else:
            raise ValueError("Cannot find a supported bilinear_type {}".format(
                self.bilinear_type))

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)

    def field_wise_func(self, field_input_1, field_input_2, n_fields):
        """
        :param field_input_1: 3d tensor, (batch, len, dim)
        :param field_input_2: 3d tensor, (batch, len, dim)
        :param n_fields: int
        :return:
        """
        import itertools
        bz = field_input_1.shape[0]

        left, right = [], []
        for i, j in itertools.combinations(list(range(n_fields)), 2):
            left.append(i)
            right.append(j)

        left_t = torch.unsqueeze(torch.tensor(left), 0).repeat(bz, 1)
        left_t = torch.unsqueeze(left_t, 2)
        right_t = torch.unsqueeze(torch.tensor(right), 0).repeat(bz, 1)
        right_t = torch.unsqueeze(right_t, 2)

        # print(f"field_input_1.shape is {field_input_1}")
        # print(f"left_t.shape is {left_t}")

        emb_left = torch.gather(field_input_1, dim=1, index=left_t)
        emb_right = torch.gather(field_input_2, dim=1, index=right_t)
        # print(f"emb_left.shape is {emb_left}")

        emb_prob = torch.mul(emb_left, emb_right)
        return emb_prob

    def forward(self, inputs):
        """
        Args:
            inputs: (batch, seq_len, dim)
        returns: (batch, seq_len, 1)
        """
        seq_len = inputs.shape[1]
        if self.bilinear_type == 'field_all':
            attn_weight = torch.matmul(inputs, self.weight)
            output = self.field_wise_func(attn_weight, inputs, seq_len)
        else:
            raise NotImplementedError

        return output


class ParallelDNNLayer(nn.Module):
    """
      Model: The Parallel Multi Layer Perceptron

      Developer: Haowen Wang

      Input shape
          - nD tensor with shape: ``(batch_size, ..., input_dim)``.
          - The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.

      Output shape
         3d tensor, (batch, n_experts, hidden_units[-1])
      """

    def __init__(self,
                 in_shape,
                 hidden_units,
                 activation='relu',
                 l2_reg=0,
                 dropout_rate=0,
                 use_bn=False,
                 apply_final_act=True,
                 seed=1024,
                 n_experts=2):
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
        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        self.apply_final_act = apply_final_act
        self.n_experts = n_experts
        self.in_shape = in_shape
        super(ParallelDNNLayer, self).__init__()
        self.build()
        self.reset_parameters()

    def build(self):
        hidden_units = [int(self.in_shape)] + list(self.hidden_units)

        self.kernels = [
            nn.Parameter(
                torch.empty(
                    (hidden_units[i], hidden_units[i + 1], self.n_experts)))
            for i in range(len(self.hidden_units))
        ]
        self.bias = [
            nn.Parameter(torch.empty((self.n_experts, self.hidden_units[i])))
            for i in range(len(self.hidden_units))
        ]

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
        for i, val in enumerate(self.kernels):
            nn.init.xavier_normal_(val)
        for i, val in enumerate(self.bias):
            nn.init.zeros_(val)

    def forward(self, inputs):
        deep_input = inputs
        deep_input = torch.unsqueeze(deep_input, dim=1)
        deep_input = deep_input.repeat((1, self.n_experts, 1))
        # (batch, experts, dim)

        for i in range(len(self.hidden_units)):
            fc = torch.torch.einsum(
                'aec,cde->aed', deep_input, self.kernels[i])
            fc += self.bias[i]

            if self.use_bn:
                fc = fc.permute((0, 2, 1))
                fc = self.bn_layers[i](fc)
                fc = fc.permute((0, 2, 1))

            if i < len(self.hidden_units) - 1 or self.apply_final_act:
                fc = F.relu(fc)

            if self.dropout_rate is not None and self.dropout_rate > 0:
                fc = self.dropout_layers[i](fc)
            deep_input = fc

        return deep_input
