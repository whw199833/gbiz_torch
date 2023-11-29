# coding: utf-8
# @Author: Haowen Wang

import torch
import torch.nn as nn
import torch.nn.functional as F
from gbiz_torch.layer import DNNLayer, MaskBlockLayer


class MaskNetModel(nn.Module):
    """
      MModel: Mask Net model

      Paper: MaskNet: Introducing Feature-Wise Multiplication to CTR Ranking Models by Instance-Guided Mask

      Link: https://arxiv.org/abs/2102.07619

      Author: Zhiqiang Wang, Qingyun She, Junlin Zhang

      Developer: Haowen Wang

      inputs:
          2d tensor (batch_size, dim_1)

      outputs:
          2d tensor (batch_size, out_dim)

      """

    def __init__(self,
                 in_shape,
                 projection_in_shape,
                 n_mask_blocks=1,
                 lego='parallel',
                 hidden_size=32,
                 projection_hidden_units=[4, 1],
                 act_fn='relu',
                 l2_reg=0.001,
                 dropout_rate=0,
                 use_bn=False,
                 seed=1024,
                 apply_final_act=False):
        """
            Args:
                hidden_units: list, unit in each hidden layer
                act_fn: string, activation function
                l2_reg: float, regularization value
                dropout_rate: float, fraction of the units to dropout.
                use_bn: boolean, if True, apply BatchNormalization in each hidden layer
                seed: int, random value for initialization

            """
        super(MaskNetModel, self).__init__()
        self.n_mask_blocks = n_mask_blocks
        self.lego = lego
        self.mask_block_layer = nn.ModuleList()

        for i in range(self.n_mask_blocks):
            if self.lego == 'serial' and i > 0:
                apply_ln_emb = False
            else:
                apply_ln_emb = True
            self.mask_block_layer.append(
                MaskBlockLayer(in_shape_list=[in_shape] * 2,
                               apply_ln_emb=apply_ln_emb,
                               hidden_size=hidden_size,
                               l2_reg=l2_reg,
                               seed=seed))

        self.dnn_layer = DNNLayer(in_shape=projection_in_shape,
                                  hidden_units=projection_hidden_units,
                                  activation=act_fn,
                                  l2_reg=l2_reg,
                                  dropout_rate=dropout_rate,
                                  use_bn=use_bn,
                                  apply_final_act=apply_final_act,
                                  seed=seed)

    def forward(self, inputs, extra_input=None):
        """
        Args:
            inputs: (batch, dim1)

        Returns:
            2d tensor (batch_size, out_dim)

        """
        if self.lego == 'parallel':
            mask_block_output_list = []
            for i in range(self.n_mask_blocks):
                mask_block_output_i = self.mask_block_layer[i](
                    [inputs, inputs])
                mask_block_output_list.append(mask_block_output_i)

            mask_block_output = torch.cat(mask_block_output_list, dim=-1)

        elif self.lego == 'serial':
            mask_block_output = inputs
            for i in range(self.n_mask_blocks):
                mask_block_output = self.mask_block_layer[i](
                    [mask_block_output, inputs])

        else:
            raise NotImplementedError

        if extra_input is not None:
            if extra_input.dim() > 2:
                extra_input = torch.flatten(extra_input, start_dim=1)
            combined_input = torch.cat(
                [mask_block_output, extra_input], dim=-1)
        else:
            combined_input = mask_block_output

        output = self.dnn_layer(combined_input)

        return output
