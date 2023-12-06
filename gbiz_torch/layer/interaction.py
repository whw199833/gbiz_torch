import torch
import torch.nn as nn
import torch.nn.functional as F


class FMLayer(nn.Module):
    """
      Model: Factorization Machine

      Paper: Factorization Machine models pairwise (order-2) feature interactions without linear term and bias.

      Link: https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf

      Author: Steffen Rendle

      Developer: Haowen Wang

      Input shape
          - 3D tensor with shape: ``(batch_size, field_size, embedding_size)``.

      Output shape
          - 2D tensor with shape: ``(batch_size, dim)``.

      """

    def __init__(self, keep_dim=False):
        super(FMLayer, self).__init__()
        self.keep_dim = keep_dim
        self.build()
        self.reset_parameters()

    def build(self):
        pass

    def reset_parameters(self):
        pass

    def forward(self, inputs):
        """
        Args:
            input_shape: Shape tuple (tuple of integers) or list of shape tuples (one per output tensor of the layer).

        """
        if inputs.dim() != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" %
                (inputs.dim()))

        fm_inputs = inputs
        squared_sum = torch.pow(torch.sum(fm_inputs, dim=1, keepdim=True), 2)
        sum_squared = torch.sum(fm_inputs * fm_inputs, dim=1, keepdim=True)
        output = squared_sum - sum_squared
        if self.keep_dim:
            output = torch.squeeze(output, dim=1)
        else:
            output = torch.sum(output, dim=2, keepdim=False)

        return output


class CrossLayer(nn.Module):
    """
      Model: DCN

      Paper: Deep & Cross Network for Ad Click Predictions

      Link: https://arxiv.org/abs/1708.05123

      Author: Ruoxi Wang, Bin Fu, Gang Fu, Mingliang Wang

      Developer: Haowen Wang

      This is the CrossLayer in DCN Model

      X_{l+1} = X_0 * X_l^T * W_l + b_l + X_l

      The shape of these tensors are X_l (batch, n_dim), W_l (batch, ), b_l (batch, ).

      inputs: 2d tensor (batch_size, n_dim)

      outputs: 2d tensor (batch_size, n_dim)

      """

    def __init__(self, in_shape, n_layers=2, l2_reg=0.001, seed=1024):
        """
            Args:
                n_layers: int, number of cross layers
                l2_reg: float, regularization value

            """
        super(CrossLayer, self).__init__()
        self.n_layers = n_layers
        self.l2_reg = l2_reg
        self.seed = seed
        self.in_shape = in_shape
        self.build()
        self.reset_parameters()

    def build(self):
        self.kernels = nn.Parameter(torch.empty(
            (self.n_layers, self.in_shape, 1)))
        self.bias = nn.Parameter(torch.empty(
            (self.n_layers, self.in_shape, 1)))

    def reset_parameters(self):
        for i in range(self.kernels.shape[0]):
            nn.init.xavier_normal_(self.kernels[i])

        for i in range(self.bias.shape[0]):
            nn.init.zeros_(self.bias[i])

    def forward(self, inputs):
        """
        Args:
            inputs: 2d tensor, (batch_size, n_dim)

        Returns:
            2d tensor, (batch_size, n_dim)
            k
        """
        inputs = inputs.unsqueeze(2)
        deep_inputs = inputs
        # deep_inputs (batch, 1, d)

        for i in range(self.n_layers):
            hidden_output = torch.tensordot(deep_inputs,
                                            self.kernels[i],
                                            dims=[[1], [0]])

            hidden_output = torch.matmul(inputs, hidden_output)

            deep_inputs = hidden_output + deep_inputs + self.bias[i]

        deep_inputs = torch.sum(deep_inputs, dim=-1)
        return deep_inputs


class FieldWiseBiInterationLayer(nn.Module):
    """
      Model: FLEN Layer

      Paper: FLEN: Leveraging Field for Scalable CTR Prediction

      Link: https://arxiv.org/pdf/1911.04690.pdf

      Author: Wenqiang Chen, Lizhang Zhan, Yuanlong Ci,Chen Lin

      Developer: Haowen Wang

      inputs: list of 3d tensors

      output: 2d tensor

      """

    def __init__(self,
                 n_inputs,
                 in_shape,
                 l2_reg=0.0001,
                 use_bias=False,
                 seed=1024):
        super(FieldWiseBiInterationLayer, self).__init__()
        self.use_bias = use_bias
        self.seed = seed
        self.l2_reg = l2_reg
        self.in_shape = in_shape
        self.fields = n_inputs
        self.build()
        self.reset_parameters()

    def build(self):
        mf_size = int(self.fields * (self.fields - 1) / 2)

        self.kernel_mf = nn.Parameter(torch.empty((mf_size, 1)))
        self.kernel_fm = nn.Parameter(torch.empty((self.fields, 1)))

        if self.use_bias:
            self.bias_mf = nn.Parameter(torch.empty((1, 1)))
            self.bias_fm = nn.Parameter(torch.empty((1, 1)))

    def reset_parameters(self):
        nn.init.xavier_normal_(self.kernel_mf)
        nn.init.xavier_normal_(self.kernel_fm)
        if self.use_bias:
            nn.init.zeros_(self.bias_mf)
            nn.init.zeros_(self.bias_fm)

    def forward(self, inputs):
        """
        :param inputs: list of 3d tensors
        :return: 2d tensor
        """
        import itertools

        bz = inputs[0].shape[0]

        if inputs[0].dim() != 3:
            raise ValueError(
                'FLEN must accept list of 3d tensors, with length > 2')

        field_wise_emb_list = inputs

        # MF part
        field_wise_vectors = torch.cat([
            torch.sum(vector_i, dim=1, keepdim=True)
            for vector_i in field_wise_emb_list
        ],
            dim=1)
        # field_wise_vectors: (batch, n_fields, dim)

        left, right = [], []
        for i, j in itertools.combinations(list(range(self.fields)), 2):
            left.append(i)
            right.append(j)

        left_t = torch.unsqueeze(torch.tensor(left), 0).repeat(bz, 1)
        left_t = torch.unsqueeze(left_t, 2)
        right_t = torch.unsqueeze(torch.tensor(right), 0).repeat(bz, 1)
        right_t = torch.unsqueeze(right_t, 2)

        emb_left = torch.gather(field_wise_vectors, dim=1, index=left_t)
        emb_right = torch.gather(field_wise_vectors, dim=1, index=right_t)

        emb_prob = torch.mul(emb_left, emb_right)
        emb_mf = torch.tensordot(emb_prob, self.kernel_mf, dims=[[1], [0]])
        emb_mf = torch.squeeze(emb_mf, dim=-1)
        # print('emb_mf ', emb_mf.shape)
        if self.use_bias:
            # print(f"emb_mf is {emb_mf.shape}, bias_mf is {self.bias_mf.shape}")
            emb_mf = torch.add(emb_mf, self.bias_mf)

        # FM part
        squared_sum = [
            torch.pow(torch.sum(vector_i, dim=1, keepdim=True), 2)
            for vector_i in field_wise_emb_list
        ]
        sum_squared = [
            torch.sum(torch.mul(vector_i, vector_i), dim=1, keepdim=True)
            for vector_i in field_wise_emb_list
        ]

        field_emb_fm = [
            square_of_sum_i - sum_of_square_i
            for square_of_sum_i, sum_of_square_i in zip(squared_sum, sum_squared)
        ]

        field_emb_fm = torch.cat(field_emb_fm, dim=1)

        emb_fm = torch.tensordot(field_emb_fm, self.kernel_fm, dims=[[1], [0]])
        emb_fm = torch.squeeze(emb_fm, dim=-1)
        # print('emb_fm ', emb_fm.shape)
        if self.use_bias:
            emb_fm = torch.add(emb_fm, self.bias_fm)

        output = torch.add(emb_mf, emb_fm)
        return output
