# coding: utf-8
# @Author: Haowen Wang

import torch
import torch.nn as nn
import torch.nn.functional as F
from gbiz_torch.layer import DNNLayer, FieldWiseBiInterationLayer


class FLENModel(nn.Module):
    """
      Model: FLEN Model

      Paper: FLEN: Leveraging Field for Scalable CTR Prediction

      Link: https://arxiv.org/pdf/1911.04690.pdf

      Author: Wenqiang Chen, Lizhang Zhan, Yuanlong Ci,Chen Lin

      Developer: Haowen Wang

      inputs:
          inputs: list of 3d tensor (batch_size, n_field, dim_1), deep features
          extra_input: 2d tensor (batch_size, dim_2), wide features

      rturns:
              2d tensor (batch_size, out_dim)

      """

    def __init__(self,
                 fields,
                 in_shape,
                 dnn_in_shape,
                 projection_in_shape,
                 hidden_units=[16],
                 projection_hidden_units=[1],
                 apply_final_act=False,
                 act_fn='relu',
                 l2_reg=0.001,
                 dropout_rate=0,
                 use_bn=False,
                 use_bias=True,
                 seed=1024):
        """
            Args:
                hidden_units: list, unit in each hidden layer
                hidden_units: list, unit in final projection layer
                act_fn: string, activation function
                l2_reg: float, regularization value
                dropout_rate: float, fraction of the units to dropout.
                use_bn: boolean, if True, apply BatchNormalization in each hidden layer
                seed: int, random value for initialization

            """
        super(FLENModel, self).__init__()
        self.flen_layer = FieldWiseBiInterationLayer(fields,
                                                     in_shape,
                                                     l2_reg=l2_reg,
                                                     use_bias=use_bias,
                                                     seed=seed)
        self.dnn_layer = DNNLayer(dnn_in_shape,
                                  hidden_units=hidden_units,
                                  activation=act_fn,
                                  l2_reg=l2_reg,
                                  dropout_rate=dropout_rate,
                                  use_bn=use_bn,
                                  seed=seed)
        self.projection_layer = DNNLayer(in_shape=projection_in_shape,
                                         hidden_units=projection_hidden_units,
                                         activation=act_fn,
                                         l2_reg=l2_reg,
                                         dropout_rate=dropout_rate,
                                         use_bn=use_bn,
                                         seed=seed,
                                         apply_final_act=apply_final_act)

    def forward(self, inputs, extra_input=None):
        """
            Args:
                inputs: list of 3d tensor: [(batch_size, n_field, dim_1), ], deep features
                extra_input: 2d tensor (batch_size, dim_2), wide features

            Returns:
                2d tensor (batch_size, out_dim)

            """
        fm_mf_output = self.flen_layer(inputs)
        # print(f"fm_mf_output is {fm_mf_output.shape}")

        concat_input = torch.cat(inputs, dim=1)
        concat_input = torch.flatten(concat_input, start_dim=1)

        dnn_output = self.dnn_layer(concat_input)

        if extra_input is None:
            combined_output = torch.cat([fm_mf_output, dnn_output], dim=-1)
        else:
            combined_output = torch.cat([fm_mf_output, dnn_output, extra_input],
                                        dim=-1)

        flen_output = self.projection_layer(combined_output)
        return flen_output
