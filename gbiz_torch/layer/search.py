import torch
import torch.nn as nn
import torch.nn.functional as F


class FineSeqLayer(nn.Module):
    """
    Model: 

    Paper: 

    Link: 

    Developer: Haowen Wang

    inputs: list of tensors [input_1, input_2]
            input_1: (batch, seq_len, dim_1)
            input_2: (batch, dim_2)

    output: (batch, seq_len, hidden_units)
    """

    def __init__(self,
                 in_shape_list,
                 n_layers=2,
                 l2_reg=0.001,
                 seed=1024,
                 drop_prob=0.0):
        """
        Args:
            n_layers: int, number of layers
            l2_reg: float
        """
        super(FineSeqLayer, self).__init__()
        self.n_layers = n_layers
        self.l2_reg = l2_reg
        self.seed = seed
        self.in_shape_list = in_shape_list
        self.drop_prob = drop_prob
        self.build()
        self.reset_parameters()

    def build(self):
        self.linear_layers = nn.ModuleList([
            nn.Linear(self.in_shape_list[1], self.in_shape_list[0])
            for i in range(self.n_layers)
        ])
        self.ln_hid_layers = nn.ModuleList(
            [nn.LayerNorm(self.in_shape_list[0]) for i in range(self.n_layers)])

        self.drop_path = nn.Dropout(self.drop_prob)

    def reset_parameters(self):
        pass

    def forward(self, inputs):
        """
        Args:
            inputs: list of tensors [seq_inputs, inputs]
            seq_inputs: (batch_size, len, dim1)
            inputs: (batch_size, dim2)
        returns:
            2d tensor (batch_size, n_dim)
        """
        seq_input, item_input = inputs

        item_input_3d = torch.unsqueeze(item_input, dim=1)
        hidden_seq_input = seq_input

        for i in range(self.n_layers):
            seq_mask_hidden_1 = self.linear_layers[i](item_input_3d)
            # seq_mask_hidden_2 = torch.tensordot(hidden_seq_input,
            #                                     seq_mask_hidden_1,
            #                                     dims=[[2], [2]])
            seq_mask_hidden_2 = torch.einsum('abc,adc->abd',
                                             (hidden_seq_input, seq_mask_hidden_1))
            seq_mask_hidden_2 = torch.sigmoid(seq_mask_hidden_2)
            V_masked = torch.mul(seq_mask_hidden_2, hidden_seq_input)
            # V_masked = torch.einsum('abd,abc->adc',
            #                         (seq_mask_hidden_2, hidden_seq_input))
            hidden_seq_input = torch.add(
                self.drop_path(V_masked), hidden_seq_input)
            hidden_seq_input = self.ln_hid_layers[i](hidden_seq_input)

        return hidden_seq_input


# class StarTopologyFCNLayer(nn.Module):
#   """
#     Model: Star Topology FCN Layer

#     Paper: One Model to Serve All: Star Topology Adaptive Recommender for Multi-Domain CTR Prediction

#     Link: https://arxiv.org/abs/2101.11427

#     Author: Xiang-Rong Sheng, Liqin Zhao, Guorui Zhou, Xinyao Ding, Binding Dai, Qiang Luo, Siran Yang, Jingshan Lv, Chi Zhang, Hongbo Deng, Xiaoqiang Zhu

#     Developer: Haowen Wang

#     Date: 2021-08-17

#     inputs: 2d tensor (batch_size, n_dim)

#     outputs: list of 2d tensor [input_1, input_2, ....]
#              input_x: (batch_size, 1)

#   """
#   def __init__(self,
#                in_shape,
#                hidden_units=[16, 8],
#                num_domains=2,
#                l2_reg=0.001,
#                seed=1024,
#                apply_final_act=False,
#                device=None,
#                dtype=None):
#     """
#     Args:
#         n_layers: int, number of layers
#         l2_reg: float
#     """
#     super(StarTopologyFCNLayer, self).__init__()
#     self.hidden_units = hidden_units
#     self.l2_reg = l2_reg
#     self.in_shape = in_shape
#     self.seed = seed
#     self.num_domains = num_domains
#     self.n_layers = len(hidden_units)
#     self.apply_final_act = apply_final_act
#     self.factory_kwargs = {'device': device, 'dtype': dtype}
#     self.build()
#     self.reset_parameters()

#   def build(self):
#     hidden_units = [self.in_shape] + self.hidden_units

#     self.shared_fc_layers, self.domain_fc_layers = nn.ModuleList(
#     ), nn.ModuleList()

#     for i in range(self.n_layers):
#       self.shared_fc_layers.append(
#           nn.Linear(hidden_units[i], hidden_units[i + 1],
#                     **self.factory_kwargs))
#       self.domain_fc_layers.append(
#           nn.ModuleList([
#               nn.Linear(hidden_units[i], hidden_units[i + 1],
#                         **self.factory_kwargs) for j in range(self.num_domains)
#           ]))

#   def reset_parameters(self):
#     pass

#   def call(self, inputs, domain_index=0):
#     """
#     Args:
#         inputs: list of 2d tensors from multiple domains, each with shape of (batch, dim2)
#         # inputs: (batch, dim2)
#         domain_index: deprecated
#     returns:
#         tensor (batch_size, n_dim)
#     """

#     domain_outputs = inputs

#     for i in range(self.n_layers):
#       domain_w = tf.keras.layers.Lambda(lambda x: tf.multiply(x[0], x[1]))([
#           self.shared_fc_weights[i],
#           self.domain_specific_weights[i][domain_index]
#       ])
#       domain_b = tf.keras.layers.Lambda(lambda x: tf.add(x[0], x[1]))(
#           [self.shared_fc_bias[i], self.domain_specific_bias[i][domain_index]])
#       domain_outputs = tf.keras.layers.Lambda(
#           lambda x: tf.keras.backend.bias_add(tf.keras.backend.dot(x[0], x[
#               1]), x[2]))([domain_outputs, domain_w, domain_b])
#       if i < self.n_layers - 1 or self.apply_final_act:
#         domain_outputs = tf.keras.layers.Lambda(lambda x: tf.nn.relu(x))(
#             domain_outputs)

#     return domain_outputs


class BridgeLayer(nn.Module):
    """
    Model: Bridge Module

    Paper: Enhancing Explicit and Implicit Feature Interactions via Information Sharing for Parallel Deep CTR Models

    Link: https://dl.acm.org/doi/abs/10.1145/3459637.3481915

    Author: Chen, Bo and Wang, Yichao and Liu, Zhirong and Tang, Ruiming and Guo, Wei and Zheng, Hongkun and Yao, Weiwei and Zhang, Muyu and He, Xiuqiang

    Developer: Haowen Wang

    inputs: 2d tensor (batch_size, n_dim)

    outputs: 2d tensor (batch_size, n_dim)

    """

    def __init__(self, in_shape, n_layers=2, l2_reg=0.001, seed=1024):
        """
        Args:
            n_layers: int, number of cross layers
            l2_reg: float, regularization value

        """
        super(BridgeLayer, self).__init__()
        self.n_layers = n_layers
        self.l2_reg = l2_reg
        self.seed = seed
        self.in_shape = in_shape
        self.build()
        self.reset_parameters()

    def build(self):
        self.deep_layers = nn.ModuleList([
            nn.Linear(self.in_shape, self.in_shape) for i in range(self.n_layers)
        ])

        self.cross_weights = nn.Parameter(
            torch.empty((self.n_layers, self.in_shape, 1)))

    def reset_parameters(self):
        for name, val in self.deep_layers.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(val)

        nn.init.xavier_normal_(self.cross_weights)

    def forward(self, inputs):
        """
        Args:
            inputs: 2d tensor, (batch_size, n_dim)

        Returns:
            2d tensor, (batch_size, n_dim)
        """
        # inputs = torch.unsqueeze(inputs, dim=1)
        bridge_inputs = inputs
        # bridge_inputs (batch, 1, d)

        for i in range(self.n_layers):
            # cross layer
            hidden_output = torch.matmul(bridge_inputs, self.cross_weights[i])
            # hidden_output = torch.einsum('abc,adc->abd',
            #                              (bridge_inputs, self.cross_weights[i]))
            # hidden_output = torch.tensordot(bridge_inputs,
            #                                 self.cross_weights[i],
            #                                 dims=[[1], [0]])
            hidden_output = torch.mul(inputs, hidden_output)
            cross_input = hidden_output + bridge_inputs

            # cross_input = tf.keras.backend.batch_dot(tf.keras.backend.dot(
            #     bridge_inputs, self.cross_weights[i]),
            #                                          inputs,
            #                                          axes=[1, 1]) + bridge_inputs
            # cross_input = tf.keras.backend.bias_add(cross_input, self.cross_bias[i])

            # deep layer
            deep_input = self.deep_layers[i](bridge_inputs)
            deep_input = F.relu(deep_input)

            bridge_inputs = torch.mul(cross_input, deep_input)

        output = torch.squeeze(bridge_inputs, dim=1)
        return output


class DCAPLayer(nn.Module):
    """
    Model: Deep Cross Attentional Product Network

    Paper: DCAP: Deep Cross Attentional Product Network for User Response Prediction

    Link: https://arxiv.org/abs/2105.08649

    Author: Zekai Chen, Fangtian Zhong, Zhumin Chen, Xiao Zhang, Robert Pless, Xiuzhen Cheng

    Developer: Haowen Wang

    inputs: 3d tensor (batch_size, seq_len, n_dim)

    outputs: 2d tensor (batch_size, n_dim)

    """

    def __init__(self,
                 fields,
                 in_shape,
                 hidden_units=16,
                 heads=4,
                 l2_reg=0.001,
                 dropout_rate=0.1,
                 seed=1024,
                 list_tag=False,
                 mha_type='origin',
                 dim_e=None,
                 synthesizer_type='dense',
                 window=1,
                 concat_heads=True,
                 pool_size=1,
                 strides=1,
                 product_type='inner',
                 n_layers=1):
        """
        Args:
            n_layers: int, number of cross layers
            l2_reg: float, regularization value
            product_type: string, inner or outer
        """
        super(DCAPLayer, self).__init__()
        self.hidden_units = hidden_units
        self.heads = heads
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.list_tag = list_tag
        self.mha_type = mha_type
        self.dim_e = dim_e
        self.synthesizer_type = synthesizer_type
        self.window = window
        self.concat_heads = concat_heads
        self.n_layers = n_layers
        self.pool_size = pool_size
        self.strides = strides
        self.fields = fields
        self.in_shape = in_shape
        self.product_type = product_type
        self.build()
        self.reset_parameters()

    def build(self):
        self.output_dim = self.fields * (self.fields - 1) * self.n_layers // 2
        from gbiz_torch.layer import MultiHeadAttentionLayer

        self.mha_layers = nn.ModuleList([
            MultiHeadAttentionLayer(in_shape=self.in_shape,
                                    hidden_units=self.hidden_units,
                                    heads=self.heads,
                                    l2_reg=self.l2_reg,
                                    dropout_rate=self.dropout_rate,
                                    seed=self.seed,
                                    mha_type=self.mha_type,
                                    dim_e=self.dim_e,
                                    synthesizer_type=self.synthesizer_type,
                                    window=self.window)
            for i in range(self.n_layers)
        ])

        self.pooling_layers = nn.ModuleList([
            nn.AvgPool1d(kernel_size=self.pool_size,
                         stride=self.strides,
                         padding=0) for i in range(self.n_layers)
        ])

    def reset_parameters(self):
        for name, val in self.mha_layers.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(val)

        for name, val in self.pooling_layers.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(val)

    def inner_product_fn(self, input_a, input_b):
        return torch.mul(input_a, input_b)

    def outer_product_fn(self, input_a, input_b):
        input_a_sum = torch.sum(torch.unsqueeze(input_a, dim=1),
                                dim=-1,
                                keepdims=False)
        input_a_sum = torch.permute(input_a_sum, (0, 2, 1))
        # input_a_sum: (batch, fields, 1)
        return torch.mul(input_a_sum, input_b)

    def enumerate_fields_fn(self):
        import itertools

        left, right = [], []
        for i in range(self.fields - 1):
            for j in range(i + 1, self.fields):
                left.append(i)
                right.append(j)

        return left, right

    def forward(self, inputs, mask=None):
        """
        Args:
            inputs: 3d tensor (batch_size, seq_len, n_dim)

        Returns:
            2d tensor, (batch_size, n_dim)
        """
        x = inputs
        outputs = []
        bz = inputs.shape[0]

        left, right = self.enumerate_fields_fn()

        left_t = torch.unsqueeze(torch.tensor(left), 0).repeat(bz, 1)
        left_t = torch.unsqueeze(left_t, 2)
        right_t = torch.unsqueeze(torch.tensor(right), 0).repeat(bz, 1)
        right_t = torch.unsqueeze(right_t, 2)

        for i in range(self.n_layers):
            z = self.mha_layers[i](x, mask=mask)
            z_pol = torch.gather(z, dim=1, index=left_t)
            x_pol = torch.gather(inputs, dim=1, index=right_t)

            if self.product_type == 'inner':
                p = self.inner_product_fn(z_pol, x_pol)
            elif self.product_type == 'outer':
                p = self.outer_product_fn(z_pol, x_pol)
            else:
                raise NotImplementedError

            outputs.append(torch.sum(p, dim=-1))

            x = self.pooling_layers[i](p)

        output = torch.cat(outputs)
        return output
