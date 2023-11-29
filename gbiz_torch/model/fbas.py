# coding: utf-8
# @Author: Haowen Wang

import torch
import torch.nn as nn
import torch.nn.functional as F
from gbiz_torch.layer import DNNLayer, WeightedSeqAggLayer, FineSeqLayer
from gbiz_torch.model import MaskNetModel


class FBASModel(nn.Module):
    """
      Model: FBAS Model

      Paper: FBAS: Field-wise User Behavior Modelling with Augmented Semantics forClick-Through Rate Prediction

      Link: 

      Developer: Haowen Wang

      inputs: (batch, seq_len, hidden_units)

      output: (batch, hidden_units)

      """

    def __init__(self,
                 in_shape_list,
                 projection_in_shape,
                 hidden_size=32,
                 projection_hidden_units=[8, 1],
                 l2_reg=0.001,
                 use_bn=False,
                 seed=1024,
                 drop_prob=0.1,
                 n_layers=2,
                 n_mask_blocks=1,
                 lego='parallel'):
        """
            Args:
                hidden_units: list, unit in each hidden layer
                act_fn: string, activation function
                l2_reg: float, regularization value
                dropout_rate: float, fraction of the units to dropout.
                use_bn: boolean, if True, apply BatchNormalization in each hidden layer
                seed: int, random value for initialization

            """
        super(FBASModel, self).__init__()

        self.field_wise_inter_layer = FineSeqLayer(in_shape_list=[in_shape_list[1], in_shape_list[2]],
                                                   n_layers=n_layers,
                                                   l2_reg=l2_reg,
                                                   seed=seed,
                                                   drop_prob=drop_prob)
        self.sequence_agg_layer = WeightedSeqAggLayer(in_shape_list=[in_shape_list[1]],
                                                      n_inputs=1,
                                                      l2_reg=l2_reg,
                                                      seed=seed)

        self.projection_layer = MaskNetModel(
            in_shape=projection_in_shape,
            projection_in_shape=projection_in_shape,
            n_mask_blocks=n_mask_blocks,
            lego=lego,
            hidden_size=hidden_size,
            projection_hidden_units=projection_hidden_units)

    def forward(self, inputs, extra_input=None):
        """
            Args:
                inputs: list, [user_feature, seq_input, item_feature]
                user_feature: (batch, dim1)
                seq_input: (batch_size, seq_len, dim_2)
                item_feature: (batch, dim3)

            Returns:
                2d tensor (batch_size, out_dim)

            """
        user_feature, seq_input, item_feature = inputs

        # find seq cross layer
        seq_output = self.field_wise_inter_layer([seq_input, item_feature])
        # print(f"seq_output.shape is {seq_output.shape}")
        seq_output = self.sequence_agg_layer(seq_output)

        if extra_input is not None:
            combined_input = torch.cat(
                [seq_output, extra_input, user_feature, item_feature], dim=-1)
        else:
            combined_input = torch.cat([seq_output, user_feature, item_feature],
                                       dim=-1)

        output = self.projection_layer(combined_input)
        return output
