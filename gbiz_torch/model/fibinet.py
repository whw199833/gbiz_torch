# coding: utf-8
# @Author: Haowen Wang

import torch
import torch.nn as nn
import torch.nn.functional as F
from gbiz_torch.layer import DNNLayer, BiLinearInteractionLayer, SENETLayer


class FiBiNetModel(nn.Module):
    """
      Model: FiBiNet Model

      PPaper: FiBiNET- Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction

      Link: https://arxiv.org/abs/1905.09433

      Author: Tongwen Huang, Zhiqi Zhang, Junlin Zhang

      Developer: Haowen Wang

      inputs: 3d tensor (batch_size, fields, n_dim)

      outputs: 2d tensor (batch_size, out_dim)

      """

    def __init__(self,
                 fields,
                 in_shape,
                 reduction_ratio,
                 projection_in_shape,
                 bilinear_type='field_all',
                 projection_hidden_units=[4, 1],
                 act_fn='relu',
                 l2_reg=0.001,
                 dropout_rate=0,
                 use_bn=False,
                 apply_final_act=False,
                 seed=1024):
        """
            Args:
                reduction_ratio: int
                bilinear_type: string, 'field_all', 'field_each', 'field_interaction'
                projection_hidden_units: list, unit in each hidden layer
                act_fn: string, activation function
                l2_reg: float, regularization value
                dropout_rate: float, fraction of the units to dropout.
                use_bn: boolean, if True, apply BatchNormalization in each hidden layer
                seed: int, random value for initialization

            """
        super(FiBiNetModel, self).__init__()
        self.senet_layer = SENETLayer(fields,
                                      reduction_ratio=reduction_ratio,
                                      act_fn=act_fn,
                                      l2_reg=l2_reg,
                                      seed=seed)

        self.emb_bilinar_layer = BiLinearInteractionLayer(
            in_shape, bilinear_type=bilinear_type, l2_reg=l2_reg, seed=seed)

        self.hidden_bilinar_layer = BiLinearInteractionLayer(
            in_shape, bilinear_type=bilinear_type, l2_reg=l2_reg, seed=seed)

        self.dnn_layer = DNNLayer(in_shape=projection_in_shape,
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
            inputs: 3d tensor (batch_size, fields, n_dim)

        Returns:
            2d tensor (batch_size, out_dim)

        """
        senet_output = self.senet_layer(inputs)

        emb_bilinear_output = self.emb_bilinar_layer(inputs)

        senet_bilinear_output = self.hidden_bilinar_layer(senet_output)

        combined_input = torch.cat([emb_bilinear_output, senet_bilinear_output],
                                   dim=-1)
        if combined_input.dim() > 2:
            combined_input = torch.flatten(combined_input, start_dim=1)

        if extra_input is not None:
            combined_input = torch.cat([combined_input, extra_input], dim=-1)
        print(f"combined_input.shape is {combined_input.shape}")
        dnn_output = self.dnn_layer(combined_input)
        return dnn_output
