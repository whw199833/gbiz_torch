import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionLayer(nn.Module):
    """
      Model: Multi-Head Attention

      Paper: Attention Is All You Need

      Link: https://arxiv.org/abs/1706.03762

      Author: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin

      Developer: Haowen Wang

      This is the multi-head self attention module in bert/transformer, depending on self attenton layer

      inputs: 3d tensor, (batch_size, fields, hidden_units)

      outputs: 3d tensor, (batch_size, fields, hidden_units)

    """

    def __init__(self,
                 in_shape,
                 hidden_units=16,
                 heads=4,
                 l2_reg=0.001,
                 dropout_rate=0.1,
                 seed=1024,
                 mha_type='origin',
                 dim_e=None,
                 synthesizer_type='dense',
                 inf=1e9,
                 window=1):
        """
            Args:
                hidden_units: int, the last dim of the inputs
                heads: int, num of self-attention modules, must be dividable by hidden_units
                l2_reg: float, regularization value
                dropout_rate: float, dropout rate
                seed: float, random seed
                kwargs: other arguments

        """
        super(MultiHeadAttentionLayer, self).__init__()
        assert (hidden_units %
                heads == 0), ('hidden_units must be dividable by heads')
        self.d_k = hidden_units // heads
        self.heads = heads
        self.hidden_units = hidden_units
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.inputs_type = 'tensor'
        self.seed = seed
        self.in_shape = in_shape

        self.dim_e = dim_e
        self.inf = inf
        self.synthesizer_type = synthesizer_type
        self.mha_type = mha_type
        self.window = window
        assert hidden_units % heads == 0, "hidden_units should be devided by heads"

        self.build()
        self.reset_parameters()

    def build(self):
        self.linear_layers = nn.ModuleList(
            [nn.Linear(self.in_shape, self.hidden_units) for _ in range(3)])
        self.linear_layers.append(
            nn.Linear(self.hidden_units, self.hidden_units))
        self.attnlayer = SelfAttentionLayer(dropout_rate=self.dropout_rate)

    def reset_parameters(self):
        for name, val in self.linear_layers.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(val)

    def forward(self, inputs, mask=None):
        """
        Args:
            inputs: (batch, seq_len, dim)

        Return:
            (batch, seq_len, hidden_units)

        """
        # print('inputs ', inputs.dtype)
        # print('inputs ', inputs.shape)
        seq_len = inputs.shape[1]
        bz = inputs.shape[0]

        query, key, value = [
            torch.reshape(self.linear_layers[i](inputs),
                          (bz, seq_len, self.heads, self.d_k)) for i in range(3)
        ]

        output = self.attnlayer([query, key, value], mask=mask)
        output = self.linear_layers[-1](output)

        return output


class SelfAttentionLayer(nn.Module):
    """
      Model: Self Attention

      Paper: Attention Is All You Need

      Link: https://arxiv.org/abs/1706.03762

      Author: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin

      Developer: Haowen Wang

      This is the scaled dot self-attention part in bert/transformer.

      inputs: list with length of 3, [query, key, value]
          query: (batch, q_len, heads, d_k)
          key: (batch, k_len, heads, d_k)
          value: (batch, k_len, heads, d_k)

      return: 3d tensor, (batch, q_len, heads*d_k), hereby heads*d_k = hidden_units

    """

    def __init__(self, dropout_rate=0.1):
        """
        Args:
            dropout_rate: float, dropout rate
        """
        super(SelfAttentionLayer, self).__init__()
        self.dropout_rate = dropout_rate
        self.build()
        self.reset_parameters()

    def build(self):
        """
        Args:
            input_shape: Shape tuple (tuple of integers) or list of shape tuples (one per output tensor of the layer).

        """
        self.dropout = nn.Dropout(self.dropout_rate)

    def reset_parameters(self):
        pass

    def forward(self, inputs, mask=None):
        """
        Args:
            inputs: list of [query, key, value]
            query: (batch, query_len, heads, d_k)
            key: (batch, key_len, heads, d_k)
            value: (batch, key_len, heads, d_k)

        Returns:
            (batch, seq_len, heads*d_k)

        """
        query, key, value = inputs

        query_len = query.shape[1]
        heads = query.shape[2]
        d_k = query.shape[3]
        query = query.permute((0, 2, 1, 3))
        key = key.permute((0, 2, 3, 1))
        # query = torch.permute(query, (0, 2, 1, 3))
        # key = torch.permute(key, (0, 2, 3, 1))

        # scores = torch.tensordot(query, key, dims=[[3], [2]])
        scores = torch.matmul(query, key)
        scores = torch.div(scores, torch.sqrt(torch.tensor(1.0 * d_k)))
        # print(f"scores.shape is {scores.shape}")
        if mask is not None:
            scores += torch.mul(mask, -1.e9)

        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        # scores: (batch, heads, q_len, k_len)
        # print('scores ', scores.shape)

        # value = torch.permute(value, (0, 2, 1, 3))
        value = value.permute((0, 2, 1, 3))
        # value: (batch, heads, v_len, dim)
        # print('value ', value.shape)

        # attn_weight = torch.tensordot(scores, value, dims=[[3], [2]])
        attn_weight = torch.matmul(scores, value)
        attn_weight = torch.reshape(attn_weight, (-1, query_len, heads * d_k))

        return attn_weight


class PositionWiseFeedForwardLayer(nn.Module):
    """
      Model: Position-wise Feed Forward layer

      Paper: Attention Is All You Need

      Link: https://arxiv.org/abs/1706.03762

      Author: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin

      Developer: Haowen Wang

      This is position-wise feed-forward networks.

      .. code-block:: python

          F = LeakyRelu(S'*W + b)*w + b)
          #F = LayerNorm(S' + Dropout()

      inputs: (batch, seq_len, in_shape)

      returns: (batch, seq_len, hidden_units)

      """

    def __init__(self,
                 in_shape,
                 intermediate_size,
                 hidden_units,
                 l2_reg=0.001,
                 act_fn='relu',
                 seed=1024):
        """
            Args:
                intermediate_size: int, units in dense layer, normally intermediate_size=4*hidden_units
                hidden_units: int, int, the last dim of the inputs
                l2_reg: float, regularization value
                dropout_rate: float, dropout rate
                act_fn: string, activation function
                kwargs:

            """
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.intermediate_size = intermediate_size
        self.hidden_units = hidden_units
        self.act_fn = act_fn
        self.l2_reg = l2_reg
        self.seed = seed
        self.in_shape = in_shape
        self.build()
        self.reset_parameters()

    def build(self):
        self.linear_layers = nn.ModuleList([
            nn.Linear(self.in_shape, self.intermediate_size),
            nn.Linear(self.intermediate_size, self.hidden_units)
        ])

    def reset_parameters(self):
        for name, val in self.linear_layers.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(val)

    def forward(self, inputs):
        """
        Args:
            inputs: (batch, seq_len, hidden_units)

        Returns:
            (batch, seq_len, hidden_units)

        """
        intermediate_output = self.linear_layers[0](inputs)
        intermediate_output = F.relu(intermediate_output)
        output = self.linear_layers[1](intermediate_output)

        return output


class TransformerEncoderLayer(nn.Module):
    """
      Model: One Transformer Encoder Layer

      Paper: Attention Is All You Need

      Link: https://arxiv.org/abs/1706.03762

      Author: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin

      Developer: Haowen Wang

      Date: 2019-11-30

      This is one transformer encoder layer, which includes: multiheadattention, add & norm, feedforward, add & norm layer

      inputs: (batch, seq_len, hidden_units)

      output: (batch, seq_len, hidden_units)


      """

    def __init__(self,
                 in_shape,
                 hidden_units,
                 heads=4,
                 intermediate_size=64,
                 act_fn='relu',
                 dropout_rate=0.1,
                 l2_reg=0.001,
                 seed=1024,
                 mha_type='origin',
                 dim_e=None,
                 synthesizer_type='dense',
                 inf=1e9,
                 window=1):
        """
            Args:
                hidden_units: int, int, the last dim of the inputs
                heads: int, num of self-attention modules, must be dividable by hidden_units
                intermediate_size: int, units in dense layer, normally intermediate_size=4*hidden_units
                act_fn: string, activation function
                dropout_rate: float, dropout rate
                l2_reg: float, regularization value
                kwargs:

            """
        super(TransformerEncoderLayer, self).__init__()
        self.hidden_units = hidden_units
        self.heads = heads
        self.intermediate_size = intermediate_size
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.act_fn = act_fn
        self.seed = seed
        self.in_shape = in_shape

        self.dim_e = dim_e
        self.inf = inf
        self.synthesizer_type = synthesizer_type
        self.mha_type = mha_type
        self.window = window

        assert self.hidden_units == self.in_shape, 'hidden_units should be same with in_shape'
        assert self.hidden_units % self.heads == 0, 'hidden_units should be devided by heads'

        self.build()
        self.reset_parameters()

    def build(self):
        """
          Args:
              input_shape: Shape tuple (tuple of integers) or list of shape tuples (one per output tensor of the layer).

        """
        self.pre_ln = nn.LayerNorm(self.in_shape)
        self.pre_dropout = nn.Dropout(self.dropout_rate)
        self.multiheadattention = MultiHeadAttentionLayer(
            in_shape=self.in_shape,
            hidden_units=self.hidden_units,
            heads=self.heads,
            l2_reg=self.l2_reg,
            dropout_rate=self.dropout_rate,
            seed=self.seed,
            mha_type=self.mha_type,
            dim_e=self.dim_e,
            synthesizer_type=self.synthesizer_type,
            inf=self.inf,
            window=self.window)

        self.last_ln = nn.LayerNorm(self.in_shape)
        self.last_dropout = nn.Dropout(self.dropout_rate)
        self.ffw_layer = PositionWiseFeedForwardLayer(self.in_shape,
                                                      self.intermediate_size,
                                                      self.hidden_units,
                                                      self.l2_reg, self.act_fn,
                                                      self.seed)

    def reset_parameters(self):
        pass

    def forward(self, inputs, mask=None, lego='standard'):
        """
        Args:
            inputs: (batch, seq_len, hidden_units)

        Return:
            (batch, seq_len, hidden_units)

        """
        if lego == 'standard':
            attention_output = self.multiheadattention(inputs, mask=mask)
            attention_output = self.pre_dropout(attention_output)
            attention_output = self.pre_ln(inputs + attention_output)

            layer_output = self.ffw_layer(attention_output)
            layer_output = self.last_dropout(layer_output)
            layer_output = self.last_ln(attention_output + layer_output)

        elif lego == 'preln':
            pre_ln_out1 = self.pre_ln(inputs)
            attn_output = self.multiheadattention(pre_ln_out1, mask=mask)
            attn_output = self.pre_dropout(attn_output)
            out1 = inputs + attn_output

            pre_ln_out2 = self.last_ln(out1)
            ffn_output = self.ffw_layer(
                pre_ln_out2)  # (batch_size, input_seq_len, d_model)
            ffn_output = self.last_dropout(ffn_output)
            # (batch_size, input_seq_len, d_model)
            layer_output = out1 + ffn_output

        return layer_output


class TransformerEncoder(nn.Module):
    """
      Model: Whole Transformer Encoder Layer

      Paper: Attention Is All You Need

      Link: https://arxiv.org/abs/1706.03762

      Author: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin

      Developer: Haowen Wang

      Date: 2019-11-30

      This is the whole encoder layer of bert/transformer. It may include multiple TransformerLayer

      inputs: (batch, seq_len, dim)

      return: (batch, seq_len, hidden_units) if return_all_layers is False, otherwise, return list of (batch, seq_len, hidden_units)

      """

    def __init__(self,
                 in_shape,
                 n_layers,
                 hidden_units,
                 heads=4,
                 intermediate_size=64,
                 act_fn='relu',
                 dropout_rate=0.1,
                 l2_reg=0.001,
                 return_all_layers=False,
                 seed=1024,
                 mha_type='origin',
                 dim_e=None,
                 synthesizer_type='dense',
                 inf=1e9,
                 window=1):
        """
            Args:
                n_layers: int, num of transformer encoder layers
                hidden_units: int, the last dim of the inputs
                heads: int, num of self-attention modules, must be dividable by hidden_units
                intermediate_size: int, units in dense layer, normally intermediate_size=4*hidden_units
                act_fn: string, activation function
                dropout_rate: float, dropout rate
                l2_reg: float, regularization value
                return_all_layers: boolean, if True return list of (batch, seq_len, hidden_units), otherwise return (batch, seq_len, hidden_units)
                kwargs:

            """
        super(TransformerEncoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_units = hidden_units
        self.heads = heads
        self.intermediate_size = intermediate_size
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.act_fn = act_fn
        self.return_all_layers = return_all_layers
        self.seed = seed
        self.in_shape = in_shape

        self.dim_e = dim_e
        self.inf = inf
        self.synthesizer_type = synthesizer_type
        self.mha_type = mha_type
        self.window = window
        assert self.hidden_units == self.in_shape, 'hidden_units should be same with in_shape'
        assert self.hidden_units % self.heads == 0, 'hidden_units should be devided by heads'

        self.build()
        self.reset_parameters()

    def build(self):
        """
        Args:
            input_shape: Shape tuple (tuple of integers) or list of shape tuples (one per output tensor of the layer).

        """
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(self.in_shape,
                                    self.hidden_units,
                                    self.heads,
                                    self.intermediate_size,
                                    self.act_fn,
                                    self.dropout_rate,
                                    self.l2_reg,
                                    self.seed,
                                    mha_type=self.mha_type,
                                    dim_e=self.dim_e,
                                    synthesizer_type=self.synthesizer_type,
                                    inf=self.inf,
                                    window=self.window)
            for i in range(self.n_layers)
        ])

    def reset_parameters(self):
        pass

    def forward(self, inputs, mask=None, lego='standard'):
        """
        Args:
            inputs: (batch, seq_len, hidden_units)

        Returns:
            (batch, seq_len, hidden_units) if return_all_layers is False, otherwise,
            return list of (batch, seq_len, hidden_units)

        """
        output = []
        output_i = inputs
        for i in range(self.n_layers):
            output_i = self.layers[i](output_i, mask=mask, lego=lego)

            if self.return_all_layers:
                output.append(output_i)

        if not self.return_all_layers:
            return output_i

        return output


class FuseLayer(nn.Module):
    """
    Model: Fuse Layer in DIN Model

    Paper: Deep Interest Network for Click-Through Rate Prediction

    Link: https://arxiv.org/abs/1706.06978

    Author: Guorui Zhou, Chengru Song, Xiaoqiang Zhu, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, Kun Gai

    Developer: Haowen Wang

    fuse layer, aka the local activation unit: tanh(w . [p, q, p*q, p-q] + b)
    this is the local activation unit in DIN paper

    inputs: list with lenght of 2, [input_a, input_b]
        input_a : (batch, a_len, dim)
        input_b: (batch, a_len, dim) or (batch, dim)

    outputs: (batch, a_len, output_size)

    """

    def __init__(self,
                 in_shape,
                 input_length,
                 hidden_units=[36],
                 act_fn='sigmoid',
                 l2_reg=0.001,
                 dropout_rate=0.1,
                 use_bn=False,
                 seed=1024,
                 weight_norm=False):
        """
        Args:
            input_length: int, sequence length of behaviour features
            hidden_units: list, units for the MLP layer
            act_fn: string, activation function
            l2_reg: float, regularization value
            dropout_rate: float, dropout rate
            use_bn: boolean, if True use BatchNormalization, otherwise not
            seed: int, random value
            weight_norm: boolean, if True scale the attention score
            kwargs:

        """
        super(FuseLayer, self).__init__()
        self.hidden_units = hidden_units
        self.l2_reg = l2_reg
        self.act_fn = act_fn
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.seed = seed
        self.weight_norm = weight_norm
        self.input_length = input_length
        self.in_shape = in_shape

        self.build()
        self.reset_parameters()

    def build(self):
        # input_size = self.in_shape * 4 if len(
        #     self.hidden_units) == 0 else self.hidden_units[-1]
        local_act_dim = self.in_shape * 4

        from gbiz_torch.layer import Dice
        self.local_act_fn = Dice(in_shape=self.hidden_units[-1],
                                 input_length=self.input_length)

        self.linear_layer = nn.ModuleList([
            nn.Linear(local_act_dim, self.hidden_units[-1]),
            nn.Linear(self.hidden_units[-1], 1)
        ])

    def reset_parameters(self):
        for name, val in self.linear_layer.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(val)

    def submat(self, input_a, input_b):
        """
        Args:
            input_a: 3d tensor, (batch, len, dim)
            input_b: tensor (batch, len, dim), must have the same shape as input_a

        Returns:
            3d tensor, (batch, len, 4*dim)

        """
        sub_a = torch.sub(input_a, input_b)
        mat_a = torch.mul(input_a, input_b)

        return torch.cat([input_a, input_b, sub_a, mat_a], dim=-1)

    def forward(self, inputs):
        """
        Args:
            inputs: list of tensors, (input_a, input_b)
                input_a: (batch, a_len, dim)
                input_b: (batch, a_len, dim) or (batch, dim)

        Returns:
            (batch, output_size)

        """
        input_a, input_b = inputs
        # print('input_a {}, input_b {}'.format(input_a.shape, input_b.shape))

        if input_b.dim() == 2:
            input_b = torch.unsqueeze(input_b, dim=1)

        len_a = input_a.shape[1]
        len_b = input_b.shape[1]
        assert len_a == len_b or len_b == 1, 'len_b should be equal with len_a, or 1'

        if len_a > len_b:
            input_b = input_b.repeat(1, len_a, 1)

        features = self.submat(input_a, input_b)
        # features: (batch, len_a, dim*4)
        # print('features ', features.shape)

        score = self.linear_layer[0](features)
        # print('score ', score.shape)
        score = self.local_act_fn(score)
        score = self.linear_layer[1](score)
        # score (batch, len_a, 1)

        # score = torch.permute(score, (0, 2, 1))
        score = score.permute((0, 2, 1))
        # score: (batch, 1, len_a)
        # print('score ', score.shape)

        if self.weight_norm:
            score = F.softmax(torch.mul(score, 0.5))

        output = torch.matmul(score, input_a)
        # output: (batch, 1, dim)
        output = torch.squeeze(output, dim=1)
        return output
