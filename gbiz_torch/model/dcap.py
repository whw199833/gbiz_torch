# coding: utf-8
# @Author: Haowen Wang

import torch
import torch.nn as nn
import torch.nn.functional as F
from gbiz_torch.layer import DNNLayer, DCAPLayer


class DCAPModel(nn.Module):
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
                 projection_in_shape,
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
                 n_layers=1,
                 projection_hidden_units=[4, 1],
                 apply_final_act=False,
                 use_bn=False,
                 act_fn='relu'):
        """
        Args:
            rnn_unit: int, rnn hidden units
            rnn_act_fn: string, activation function in rnn
            return_sequences: bool, whether return hidden states of rnn
            apply_final_act: whether to apply act in final layer
            projection_hidden_units: list, unit in each hidden layer
            act_fn: string, activation function
            l2_reg: float, regularization value
            dropout_rate: float, fraction of the units to dropout.
            use_bn: boolean, if True, apply BatchNormalization in each hidden layer
            seed: int, random value for initialization

        """
        super(DCAPModel, self).__init__()
        self.dcap_layer = DCAPLayer(fields,
                                    in_shape,
                                    hidden_units=hidden_units,
                                    heads=heads,
                                    l2_reg=l2_reg,
                                    dropout_rate=dropout_rate,
                                    seed=seed,
                                    list_tag=list_tag,
                                    mha_type=mha_type,
                                    dim_e=dim_e,
                                    synthesizer_type=synthesizer_type,
                                    window=window,
                                    concat_heads=concat_heads,
                                    pool_size=pool_size,
                                    strides=strides,
                                    product_type=product_type,
                                    n_layers=n_layers)

        self.dnn_layer = DNNLayer(in_shape=projection_in_shape,
                                  hidden_units=projection_hidden_units,
                                  activation=act_fn,
                                  l2_reg=l2_reg,
                                  dropout_rate=dropout_rate,
                                  use_bn=use_bn,
                                  apply_final_act=apply_final_act,
                                  seed=seed)

    def forward(self, inputs, mask=None, extra_input=None):
        """
        Args:
            inputs: 3d tensor (batch_size, len, dim_1), sequence features
            extra_input: 2d tensor (batch_size, dim_2), wide features

        Returns:
            2d tensor (batch_size, out_dim)

        """
        dcap_output = self.dcap_layer(inputs, mask=mask)

        if dcap_output.dim() > 2:
            dcap_output = torch.flatten(dcap_output, start_dim=1)

        if extra_input is not None:
            combined_input = torch.cat([dcap_output, extra_input], dim=-1)
        else:
            combined_input = dcap_output

        output = self.dnn_layer(combined_input)
        return output
