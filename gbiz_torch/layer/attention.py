import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossStitchLayer(nn.Module):
    """
      Model: Cross Stitch Layer

      Paper: Cross-stitch Networks for Multi-task Learning

      Link: https://arxiv.org/pdf/1604.03539

      Author: Ishan Misra, Abhinav Shrivastava, Abhinav Gupta, Martial Hebert

      Developer: Haowen Wang

      Input shape
          - nD tensor with shape: ``(batch_size, ..., input_dim)``.
          - The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.

      Output shape
          - nD tensor with shape: ``(batch_size, ..., hidden_units[-1])``.
          - For instance, for a 2D input with shape ``(batch_size, input_dim)``,
            the output would have shape ``(batch_size, hidden_units[-1])``.

    """

    def __init__(self, in_shape_list):
        super(CrossStitchLayer, self).__init__()
        self.in_shape_list = in_shape_list
        self.total_in_shape = sum(in_shape_list)
        self.build()
        self.reset_parameters()

    def build(self):
        """
        Args:
            input_shape: Shape tuple (tuple of integers) or list of shape tuples (one per output tensor of the layer).
        """
        self.cross_stitch_weight = nn.Parameter(
            torch.empty((self.total_in_shape, self.total_in_shape)))

    def reset_parameters(self):
        nn.init.eye_(self.cross_stitch_weight)

    def forward(self, inputs):
        """
        Args:
            inputs: list of tensors (N, ..., dim_i)
        returns:
            list of tensors (N, ..., dim_i)
        """
        if not isinstance(inputs, (list or tuple)) or len(inputs) <= 1:
            raise ValueError(
                'inputs is not list or tuple, nothing to cross stitch')

        bz = inputs[0].shape[0]
        combined_input = torch.cat(inputs, dim=-1)
        cross_stitch_output = torch.matmul(combined_input,
                                           self.cross_stitch_weight)
        print('cross_stitch_output ', cross_stitch_output.shape)
        # for now, only supports list of 2d tensors as inputs
        outputs = []
        start_index = 0
        end_index = 0
        for i in range(len(inputs)):
            start_index = end_index
            end_index += self.in_shape_list[i]
            outputs.append(
                torch.reshape(cross_stitch_output[:, start_index:end_index],
                              (bz, self.in_shape_list[i])))

        return outputs


class SENETLayer(nn.Module):
    """
      Model: Squeeze and Excitation Layer

      Paper: FiBiNET- Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction

      Link: https://arxiv.org/abs/1905.09433

      Author: Tongwen Huang, Zhiqi Zhang, Junlin Zhang

      Developer: Haowen Wang

      inputs: (batch, fields, dim)

      output: (batch, fields, dim)
      """

    def __init__(self,
                 fields,
                 reduction_ratio,
                 act_fn='relu',
                 l2_reg=0.0001,
                 seed=23):
        super(SENETLayer, self).__init__()
        self.reduction_ratio = reduction_ratio
        self.act_fn = act_fn
        self.l2_reg = l2_reg
        self.seed = seed
        self.fields = fields
        self.build()
        self.reset_parameters()

    def build(self):
        hidden_size = max(1, self.fields // self.reduction_ratio)

        self.w1 = nn.Parameter(torch.empty((self.fields, hidden_size)))
        self.w2 = nn.Parameter(torch.empty((hidden_size, self.fields)))

    def reset_parameters(self):
        nn.init.xavier_normal_(self.w1)
        nn.init.xavier_normal_(self.w2)

    def forward(self, inputs):
        """
        Args:
          inputs: (batch, fields, dim)
        returns:
            (batch, fields, dim)
        """
        z = torch.mean(inputs, dim=-1, keepdim=False)

        # squeeze and excitation part
        A = F.relu(torch.matmul(z, self.w1))
        A = F.relu(torch.matmul(A, self.w2))
        A = torch.unsqueeze(A, dim=-1)
        V = torch.mul(inputs, A)
        # V (batch, fields, dim)
        return V


def positional_signal(hidden_size,
                      length,
                      min_timescale=1.0,
                      max_timescale=1e4):
    """
      Helper function, constructing basic positional encoding.
      The code is partially based on implementation from Tensor2Tensor library
      https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
      """

    import numpy as np

    if hidden_size % 2 != 0:
        raise ValueError(
            "The hidden dimension of the model must be divisible by 2. Currently it is {hidden_size}"
        )

    position = torch.tensor(np.arange(0, length), dtype=torch.float)
    num_timescales = int(hidden_size // 2)
    log_timescale_increment = (
        np.log(float(max_timescale) / float(min_timescale)) /
        (num_timescales - 1))

    # inv_timescales = (min_timescale * tf.keras.backend.exp(
    #     tf.keras.backend.arange(num_timescales, dtype=tf.keras.backend.floatx())
    #     * -log_timescale_increment))

    rangess = np.arange(0, num_timescales) * (-log_timescale_increment)
    inv_timescales = torch.tensor(min_timescale * np.exp(rangess),
                                  dtype=torch.float)

    # inv_timescales = torch.tensor((min_timescale * torch.exp(
    #     torch.mul(
    #         torch.tensor(np.arange(0, num_timescales) * (-log_timescale_increment)))
    scaled_time = torch.mul(torch.unsqueeze(position, dim=1),
                            torch.unsqueeze(inv_timescales, dim=0))
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
    signal = torch.unsqueeze(signal, dim=0)

    return signal


def learnabel_position_encoding(hidden_size, length, seed=1024):
    import numpy as np

    position_index = np.arange(length)[np.newaxis, :]
    position_tensor = torch.LongTensor(position_index)

    embedding_table = nn.Embedding(length, hidden_size)
    position_embedding = embedding_table(position_tensor)

    return position_embedding


class PositionalEncodingLayer(nn.Module):
    """
      Model: Positional Encoding Layer

      Paper: Attention Is All You Need

      Link: https://arxiv.org/abs/1706.03762

      Author: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin

      Developer: Haowen Wang

      This is the multi-head self attention module in bert/transformer, depending on self attenton layer

      inputs: 3d tensor, (batch_size, fields, hidden_units)

      outputs: 3d tensor, (batch_size, fields, hidden_units)
      """

    def __init__(self,
                 input_length,
                 in_shape,
                 min_timescale=1.0,
                 max_timescale=1.0e4,
                 pos_type='fixed',
                 seed=1024,
                 add_pos=True,
                 hidden_size=None):
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale
        self.seed = seed
        self.pos_type = pos_type
        self.add_pos = add_pos
        self.hidden_size = hidden_size
        self.input_length = input_length
        self.in_shape = in_shape
        super(PositionalEncodingLayer, self).__init__()
        self.build()
        self.reset_parameters()

    def build(self):
        self.hidden_size = self.in_shape if self.hidden_size is None else self.hidden_size

        if self.pos_type == 'fixed':
            self.signal = positional_signal(self.hidden_size, self.input_length,
                                            self.min_timescale, self.max_timescale)
        elif self.pos_type == 'learnable':
            self.signal = learnabel_position_encoding(int(self.hidden_size),
                                                      int(self.input_length),
                                                      self.seed)

    def reset_parameters(self):
        pass

    def forward(self, inputs):
        if self.add_pos:
            bz = inputs.shape[0]
            self.signal = self.signal.repeat(bz, 1, 1)
            # print('signal ', self.signal.shape)
            output = inputs + self.signal
            # print('output ', output.shape)
            return output
        else:
            return self.signal


class HierarchicalAttnAggLayer(nn.Module):
    """
      Model: Hierarchical Attention Aggregation layer

      Paper: Interpretable Click-Through Rate Prediction through Hierarchical Attention

      Link: https://dl.acm.org/doi/pdf/10.1145/3336191.3371785

      Author: Zeyu Li, Wei Cheng, Yang Chen, Haifeng Chen, Wei wang

      Developer: Haowen Wang

      inputs: (batch, fields, dim)

      outputs: 2d tensor, (batch, dim3)

      """

    def __init__(self,
                 in_shape,
                 n_layers=4,
                 hidden_size=16,
                 l2_reg=0,
                 seed=1024):
        self.hidden_size = hidden_size
        self.l2_reg = l2_reg
        self.seed = seed
        self.n_layers = n_layers
        self.in_shape = in_shape
        super(HierarchicalAttnAggLayer, self).__init__()
        self.build()
        self.reset_parameters()

    def build(self):
        self.kernels = []
        self.contexts = []

        self.kernels = nn.ModuleList([
            nn.Linear(self.in_shape, self.hidden_size, bias=False)
            for i in range(self.n_layers)
        ])

        self.contexts = nn.ModuleList([
            nn.Linear(self.hidden_size, 1, bias=False)
            for i in range(self.n_layers)
        ])

        self.attn_kernel = nn.Linear(
            self.in_shape, self.hidden_size, bias=False)
        self.attn_context = nn.Linear(self.hidden_size, 1, bias=False)

    def reset_parameters(self):
        for name, val in self.kernels.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(val)

        for name, val in self.contexts.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(val)

        for name, val in self.attn_kernel.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(val)

        for name, val in self.attn_context.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(val)

    def forward(self, inputs):
        """
        :param inputs: 3d tensor, (batch, fields, dim)
        :param kwargs:
        :return: 2d tensor, (batch, d3)
        """
        attn_output_list = []
        hidden_representation_i = inputs
        for i in range(self.n_layers):
            hidden_output = self.kernels[i](inputs)
            hidden_output = F.relu(hidden_output)
            # hidden_output: (batch, fields, hidden_size)

            attn_weight = self.contexts[i](hidden_output)
            attn_score = F.softmax(attn_weight, dim=1)
            # attn_score: (batch, fields, 1)

            attn_score = attn_score.permute((0, 2, 1))
            u_i = torch.matmul(attn_score, hidden_representation_i)
            # u_i: (batch, 1, dim)
            attn_output_list.append(u_i)

            hidden_representation_i = torch.add(torch.mul(u_i, inputs),
                                                hidden_representation_i)

        attn_output = torch.cat(attn_output_list, dim=1)
        # attn_output: (batch, k, dim), k = self.n_layers

        attn_hidden_output = self.attn_kernel(attn_output)
        attn_hidden_output = F.relu(attn_hidden_output)
        # attn_hidden_output: (batch, k, dim)

        attn_final_weight = self.attn_context(attn_hidden_output)
        attn_final_score = F.softmax(attn_final_weight, dim=1)
        # attn_final_score: (batch, k, 1)

        attn_final_score = attn_final_score.permute((0, 2, 1))
        output = torch.matmul(attn_final_score, attn_output)
        # output: (batch, 1, dim)
        output = torch.squeeze(output, dim=1)

        return output
