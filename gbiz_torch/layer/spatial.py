import torch
import torch.nn as nn
import torch.nn.functional as F


class CoActionLayer(nn.Module):
    """
      Model: co-action Layer

      Paper: CAN: Revisiting Feature Co-Action for Click-Through Rate Prediction

      Link: https://arxiv.org/abs/2011.05625

      Author: Guorui Zhou, Weijie Bian, et.al

      Developer: Haowen Wang

      inputs: inputs: tuple, (input_a, input_b)
              input_a: (batch, seq, dim1) or (batch, dim1)
              input_b: (batch, dim2)

              dim2 >> dim1

      return: (batch, dim)
      """

    def __init__(self,
                 in_shape_list,
                 hidden_units=[8, 4],
                 apply_final_act=False,
                 use_bias=True):
        super(CoActionLayer, self).__init__()
        self.hidden_units = hidden_units
        self.apply_final_act = apply_final_act
        self.use_bias = use_bias
        self.in_shape_list = in_shape_list
        self.build()
        self.reset_parameters()

    def build(self):
        input_size = self.in_shape_list[0]
        self.hidden_size = [int(input_size)] + list(self.hidden_units)
        self.n_layers = len(self.hidden_size) - 1
        self.total_params = sum([
            (self.hidden_size[i] + 1) *
            self.hidden_size[i + 1] if self.use_bias else self.hidden_size[i] *
            self.hidden_size[i + 1] for i in range(self.n_layers)
        ])

        input_b_size = int(self.in_shape_list[1])
        assert (input_b_size >= self.total_params), (
            'input_b last unit must be at least of {}'.format(self.total_params))

    def reset_parameters(self):
        pass

    def reshape_weight_bias(self, input):
        self.kernels, self.bias = [], []
        index = 0
        for i in range(self.n_layers):
            prev_unit = self.hidden_size[i]
            last_unit = self.hidden_size[i + 1]
            weight = torch.reshape(input[:, index:index + prev_unit * last_unit],
                                   (-1, prev_unit, last_unit))
            self.kernels.append(weight)
            index += prev_unit * last_unit

            if self.use_bias:
                bias = torch.reshape(input[:, index:index + last_unit],
                                     (-1, 1, last_unit))
                self.bias.append(bias)
                index += last_unit

    def forward(self, inputs):
        """
        Args:
            inputs: tuple, (input_a, input_b)
            input_a: (batch, seq, dim1) or (batch, dim1)
            input_b: (batch, dim2)

        returns: (batch, l)
        """
        input_a, input_b = inputs

        # split input_b to get weights and bias
        self.reshape_weight_bias(input_b)

        # perform mlp
        need_reshape_back = False
        if input_a.dim() <= 2:
            hidden_output = torch.unsqueeze(input_a, dim=1)
            need_reshape_back = True
        else:
            hidden_output = input_a

        for i in range(len(self.kernels)):
            hidden_output = torch.matmul(hidden_output, self.kernels[i])
            if self.use_bias:
                hidden_output += self.bias[i]

            if i < len(self.kernels) - 1:
                hidden_output = F.selu(hidden_output)
            elif self.apply_final_act:
                hidden_output = F.selu(hidden_output)

        if need_reshape_back:
            hidden_output = torch.squeeze(hidden_output, dim=1)
        return hidden_output


class MaskBlockLayer(nn.Module):
    """
      Model: Mask Block Layer

      Paper: MaskNet: Introducing Feature-Wise Multiplication to CTR Ranking Models by Instance-Guided Mask

      Link: https://arxiv.org/abs/2102.07619

      Author: Zhiqiang Wang, Qingyun She, Junlin Zhang

      Developer: Haowen Wang

      inputs: inputs: tuple, (input_a, input_b)
              input_a: (batch, dim1)
              input_b: (batch, dim2)

      return: (batch, dim1)
      """

    def __init__(self,
                 in_shape_list,
                 apply_ln_emb=True,
                 hidden_size=32,
                 l2_reg=0.001,
                 seed=11):
        super(MaskBlockLayer, self).__init__()
        self.hidden_size = hidden_size
        self.l2_reg = l2_reg
        self.seed = seed
        self.apply_ln_emb = apply_ln_emb
        self.in_shape_list = in_shape_list
        self.build()
        self.reset_parameters()

    def build(self):
        input_size_a = self.in_shape_list[0]
        input_size = self.in_shape_list[1]

        self.se_layer = nn.ModuleList([
            nn.Linear(input_size, self.hidden_size),
            nn.Linear(self.hidden_size, input_size_a)
        ])

        if self.apply_ln_emb:
            self.ln_emb_layer = nn.LayerNorm(input_size_a)
        self.ln_hid_layer = nn.LayerNorm(input_size_a)

        self.hidden_weight = nn.Parameter(torch.empty(
            (input_size_a, input_size_a)))

    def reset_parameters(self):
        nn.init.xavier_normal_(self.hidden_weight)

        for name, val in self.se_layer.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(val)

    def apply_instance_guide_mask(self, inputs):
        """
        :param inputs:
        :return:
        """
        output_1 = self.se_layer[0](inputs)
        output = self.se_layer[1](output_1)
        return output

    def forward(self, inputs):
        """
        Args:
            inputs: tuple, (input_a, input_b)
            input_a: (batch, seq, dim1) or (batch, dim1)
            input_b: (batch, dim2)

        returns: input_a.shape
        """
        input_a, input_b = inputs

        V_mask = self.apply_instance_guide_mask(input_b)
        if self.apply_ln_emb:
            input_a = self.ln_emb_layer(input_a)

        if input_a.dim() == 3 and V_mask.dim() == 2:
            V_mask = torch.unsqueeze(V_mask, dim=1)

        V_masked = torch.mul(V_mask, input_a)
        V_hidden = self.ln_hid_layer(
            torch.matmul(V_masked, self.hidden_weight))
        V_output = F.relu(V_hidden)
        return V_output


class ContextNetBlockLayer(nn.Module):
    """
      Model: Context Embedding and Context Block Layer

      Paper: ContextNet: A Click-Through Rate Prediction Framework Using Contextual information to Refine Feature Embedding

      Link: https://arxiv.org/abs/2107.12025

      Author: Zhiqiang Wang, Qingyun She, PengTao Zhang, Junlin Zhang

      Developer: Haowen Wang

      Date: 2021-08-17

      inputs: inputs: (batch, fields, dim1)

      return: (batch, fields, dim)
      """

    def __init__(self,
                 fields,
                 in_shape,
                 hidden_size=32,
                 share_aggregation_tag=True,
                 point_fnn_tag=False,
                 contextnet_block_layers=1,
                 l2_reg=0.001,
                 seed=11):
        super(ContextNetBlockLayer, self).__init__()
        self.hidden_size = hidden_size
        self.l2_reg = l2_reg
        self.seed = seed
        self.share_aggregation_tag = share_aggregation_tag
        self.point_fnn_tag = point_fnn_tag
        self.in_shape = in_shape
        self.fields = fields
        self.contextnet_block_layers = contextnet_block_layers
        self.build()
        self.reset_parameters()

    def build(self):
        if self.share_aggregation_tag:
            self.excite_layer = nn.Linear(self.in_shape, self.hidden_size)
        else:
            self.excite_layer = nn.Linear(self.fields * self.in_shape,
                                          self.fields * self.hidden_size)

        self.se_layer = nn.Linear(self.fields * self.hidden_size,
                                  self.fields * self.in_shape)

        self.contextnet_layers = nn.ModuleList([
            nn.Linear(self.in_shape, self.in_shape)
            for i in range(self.contextnet_block_layers)
        ])

        self.ln_layers = nn.ModuleList([
            nn.LayerNorm(self.in_shape)
            for i in range(self.contextnet_block_layers)
        ])

        if self.point_fnn_tag:
            self.context_block_w2_layers = nn.ModuleList([
                nn.Linear(self.in_shape, self.in_shape, bias=False)
                for i in range(self.contextnet_block_layers)
            ])

    def reset_parameters(self):
        for name, val in self.excite_layer.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(val)

        for name, val in self.se_layer.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(val)

        for name, val in self.contextnet_layers.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(val)

        if self.point_fnn_tag:
            for name, val in self.context_block_w2_layers.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_normal_(val)

    def context_embedding_fn(self, inputs):
        """
        :param inputs: (batch, fields, dim)
        :return:
        """
        bz = inputs.shape[0]
        if self.share_aggregation_tag:
            output_1 = self.excite_layer(inputs)
            output_1 = torch.reshape(
                output_1, (bz, self.fields * self.hidden_size))
        else:
            inputs_reshaped = torch.reshape(inputs,
                                            (bz, self.fields * self.in_shape))
            output_1 = self.excite_layer(inputs_reshaped)

        context_embedding = self.se_layer(output_1)
        context_embedding = torch.reshape(context_embedding,
                                          (bz, self.fields, self.in_shape))

        return context_embedding

    def contextnet_block_fn(self, i, inputs, context_embedding):
        """
        params:
          inputs: (batch, fields, dim)
          context_embedding: (batch, fields, dim)

        outputs: (batch, fields, dim)
        """
        merge_output = torch.mul(inputs, context_embedding)
        # merge_output: (batch, fields, dim)
        hidden_output = self.contextnet_layers[i](merge_output)

        if self.point_fnn_tag:
            hidden_output = F.relu(hidden_output)
            hidden_output = self.context_block_w2_layers[i](hidden_output)
            hidden_output = hidden_output + inputs

        output = self.ln_layers[i](hidden_output)

        return output

    def forward(self, inputs):
        """
        Args:
            inputs: (batch, fields, dim1)

        returns: (batch, fields*dim1)
        """
        bz = inputs.shape[0]
        context_embedding = self.context_embedding_fn(inputs)

        contextnet_block_output = inputs
        for i in range(self.contextnet_block_layers):
            contextnet_block_output = self.contextnet_block_fn(
                i, contextnet_block_output, context_embedding)

        output = torch.reshape(contextnet_block_output,
                               (bz, self.fields * self.in_shape))
        return output


class WeightedSeqAggLayer(nn.Module):
    def __init__(self,
                 in_shape_list,
                 n_inputs=2,
                 temperature=1,
                 l2_reg=0.001,
                 seed=11):
        super(WeightedSeqAggLayer, self).__init__()
        self.l2_reg = l2_reg
        self.seed = seed
        self.n_inputs = n_inputs
        self.temperature = temperature
        self.in_shape_list = in_shape_list
        self.build()
        self.reset_parameters()

    def build(self):
        if self.n_inputs == 1:
            input_size = self.in_shape_list[0]
            output_size = 1
        elif self.n_inputs == 2:
            output_size = self.in_shape_list[0]
            input_size = self.in_shape_list[1]

        self.weight = nn.Parameter(torch.empty((input_size, output_size)))

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)

    def get_masked_score(self, attn, mask=None):
        """
            Args:
                attn: (batch, len, 1)
                mask: (batch, len, 1), masked token is labeled with 1
            """

        # print(f"attn.shape is {attn.shape}")
        if mask is not None:
            if mask.dim() < 3:
                mask = torch.unsqueeze(mask, -1)
            attn += torch.mul(mask, -1e9)

        attn = torch.div(attn, self.temperature)
        score = F.softmax(attn, dim=1)
        return score

    def forward(self, inputs, mask=None):
        """
        Args:
            inputs: (batch, seq_len, dim)
        returns: (batch, dim)
        """
        if self.n_inputs == 1:
            sequence_input = inputs
            attn = torch.matmul(sequence_input, self.weight)
            # attn: (batch, seq_len, 1)
        elif self.n_inputs == 2:
            sequence_input, hidden_state = inputs
            hidden_state = torch.matmul(hidden_state, self.weight)
            hidden_state_3d = torch.unsqueeze(hidden_state, dim=-1)
            # hidden_state_3d: (batch, dim, 1)
            attn = torch.matmul(sequence_input, hidden_state_3d)
            # attn: (batch, seq_len, 1)

        score = self.get_masked_score(attn, mask)

        # output = torch.tensordot(sequence_input, score, dims=[[1], [1]])
        output = torch.einsum('abc,abd->acd', (sequence_input, score))
        output = torch.squeeze(output, dim=-1)
        # output (batch, dim)
        return output
