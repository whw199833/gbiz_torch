# coding: utf-8
# @Author: Haowen Wang

import torch
import torch.nn as nn
import torch.nn.functional as F

from gbiz_torch.layer import TransformerEncoder, DNNLayer, PositionalEncodingLayer


class Bert4RecModel(nn.Module):
    """
      Model: Bert4Rec Model

      Paper: BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer

      Link: https://arxiv.org/abs/1904.06690

      Author: Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, Peng Jiang

      Developer: Haowen Wang

      inputs: (batch, seq_len, hidden_units)

      output: (batch, hidden_units)

      """

    def __init__(self,
                 in_shape,
                 input_length,
                 projection_in_shape,
                 hidden_units=16,
                 heads=4,
                 intermediate_size=64,
                 n_transform_layers=1,
                 act_fn='relu',
                 dropout_rate=0,
                 l2_reg=0.001,
                 return_all_layers=False,
                 projection_hidden_units=[4, 1],
                 apply_final_act=False,
                 use_bn=False,
                 seed=1024,
                 min_timescale=1.0,
                 max_timescale=1.0e4,
                 pos_type='fixed',
                 mha_type='origin',
                 dim_e=3,
                 synthesizer_type='dense',
                 inf=1e9):
        """
            Args:
                n_transform_layers: int, num of transformer encoder layers
                hidden_units: int, the last dim of the inputs
                heads: int, num of self-attention modules, must be dividable by hidden_units
                intermediate_size: int, units in dense layer, normally intermediate_size=4*hidden_units
                act_fn: string, activation function
                dropout_rate: float, dropout rate
                l2_reg: float, regularization value
                return_all_layers: boolean, if True return list of (batch, seq_len, hidden_units), otherwise return (batch, seq_len, hidden_units)
                projection_hidden_units: list, units for the projection layer
                apply_final_act: whether to apply act in final layer
                use_bn: boolean, if True use BatchNormalization, otherwise not
                seed: int, random value

            """
        super(Bert4RecModel, self).__init__()
        self.hidden_units = hidden_units
        assert (hidden_units %
                heads == 0), ('hidden_units must be dividable by heads')
        self.pos_type = pos_type

        if self.pos_type != 'no_pos':
            self.position_layer = PositionalEncodingLayer(
                input_length,
                in_shape,
                min_timescale=min_timescale,
                max_timescale=max_timescale,
                pos_type=self.pos_type)

        self.bert_layer = TransformerEncoder(in_shape=in_shape,
                                             n_layers=n_transform_layers,
                                             hidden_units=hidden_units,
                                             heads=heads,
                                             intermediate_size=intermediate_size,
                                             act_fn=act_fn,
                                             dropout_rate=dropout_rate,
                                             l2_reg=l2_reg,
                                             return_all_layers=return_all_layers,
                                             mha_type=mha_type,
                                             dim_e=dim_e,
                                             synthesizer_type=synthesizer_type,
                                             inf=inf)

        self.dnn_layer = DNNLayer(in_shape=projection_in_shape,
                                  hidden_units=projection_hidden_units,
                                  activation=act_fn,
                                  l2_reg=l2_reg,
                                  dropout_rate=dropout_rate,
                                  use_bn=use_bn,
                                  apply_final_act=apply_final_act,
                                  seed=seed)

    def forward(self, inputs, extra_input=None, lego='standard'):
        """
        Args:
            inputs: 3d tensor (batch, seq_len, hidden_units)

        Returns:
            2d tensor (batch_size, out_dim)

        """
        input_dim = inputs.shape[-1]
        assert (input_dim == self.hidden_units), (
            "last dim of inputs must equal to hidden_units")

        if self.pos_type != 'no_pos':
            seq_input = torch.mul(
                inputs, torch.sqrt(torch.tensor(input_dim, dtype=torch.float32)))
            seq_input = self.position_layer(seq_input)
        else:
            seq_input = inputs

        # print('seq_input ', seq_input.dtype)

        bert_output = self.bert_layer(seq_input, lego=lego)
        if bert_output.dim() > 2:
            bert_output = torch.flatten(bert_output, start_dim=1)

        if extra_input is not None:
            combined_input = torch.cat([bert_output, extra_input], dim=-1)
        else:
            combined_input = bert_output

        # print(f"combined_input.shape is {combined_input.shape}")
        output = self.dnn_layer(combined_input)
        return output
