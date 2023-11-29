import torch
import torch.nn as nn
import torch.nn.functional as F


class Dice(nn.Module):
    """

    Model: Dice activation function

    Paper: Deep Interest Network for Click-Through Rate Prediction

    Link: https://arxiv.org/abs/1706.06978

    Author: Guorui Zhou, Chengru Song, Xiaoqiang Zhu, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, Kun Gai

    Developer: Haowen Wang

    """

    def __init__(self, in_shape, input_length, axis=-1, epsilon=1.e-9):
        """
        Args:
            axis: int, the axis used to calculate data distribution
            epsilon: double, for smooth purpose
        """
        super(Dice, self).__init__()
        self.axis = axis
        self.in_shape = in_shape
        self.epsilon = epsilon
        self.input_length = input_length
        self.build()
        self.reset_parameters()

    def build(self):
        self.alpha = nn.Parameter(torch.empty((self.in_shape)))

        self.bn = nn.BatchNorm1d(self.input_length, eps=self.epsilon)

    def reset_parameters(self):
        nn.init.zeros_(self.alpha)

    def forward(self, inputs):
        normed_input = self.bn(inputs)
        p = torch.sigmoid(normed_input)
        # print('p ', p.shape)
        return torch.mul(p, inputs) + torch.mul(torch.mul(1.0 - p, self.alpha),
                                                inputs)


# if __name__ == "__main__":
#     test_input = torch.randn((16, 5))
#     dice_layer = Dice(
#         in_shape=test_input.shape[1], input_length=test_input.shape[0])
#     output = dice_layer(test_input)
#     print(output.shape)
#     print(output)
