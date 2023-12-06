# coding: utf-8
# @Author: Haowen Wang

import torch
import torch.nn as nn
from gbiz_torch.layer import DNNLayer, CINLayer


class xDeepFMModel(nn.Module):
    """
    Model: xDeepFM Model

    Paper: xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems

    Link: https://arxiv.org/abs/1803.05170

    Author: Jianxun Lian, Xiaohuan Zhou, Fuzheng Zhang, Zhongxia Chen, Xing Xie, Guangzhong Sun

    Developer: Haowen Wang

    inputs: 3d tensor (batch_size, fields, n_dim)

    outputs: 2d tensor (batch_size, out_dim)

    """

    def __init__(self, hidden_units, act_fn='relu', l2_reg=0.001, dropout_rate=0, use_bn=False,
                 seed=1024, cin_hidden_units=[100, 100], cin_act_fn='relu', cin_l2_reg=0.001, name='xDeepFMModel'):
        """
        Args:
            hidden_units: list, unit in each hidden layer
            act_fn: string, activation function
            l2_reg: float, regularization value
            dropout_rate: float, fraction of the units to dropout.
            use_bn: boolean, if True, apply BatchNormalization in each hidden layer
            seed: int, random value for initialization
            hidden_units: list, unit in each cin layer
            act_fn: string, activation function in cin layer
            l2_reg: float, regularization value in cin layer

        """
        super(xDeepFMModel, self).__init__(name='xDeepFMModel')
        self.cin_layer = CINLayer(hidden_units=cin_hidden_units, act_fn=cin_act_fn,
                                  l2_reg=cin_l2_reg, name="{}_cin_layer".format(name))
        self.dnn_layer = DNNLayer(hidden_units=hidden_units, activation=act_fn, l2_reg=l2_reg,
                                  dropout_rate=dropout_rate, use_bn=use_bn, seed=seed, name="{}_dnn_layer".format(name))

    def call(self, inputs, training=None):
        """
        Args:
            inputs: 3d tensor (batch_size, fields, n_dim)

        Returns:
            2d tensor (batch_size, out_dim)

        """
        cin_output = self.cin_layer(inputs)

        flat_inputs = tf.keras.layers.Flatten()(inputs)
        tf.logging.info('xDeepFMModel: flat_inputs {}'.format(flat_inputs))
        dnn_output = self.dnn_layer(flat_inputs, training=training)

        combined_output = tf.keras.layers.Concatenate()(
            [cin_output, dnn_output])
        tf.logging.info(
            'xDeepFMModel: combined_output {}'.format(combined_output))
        return combined_output
