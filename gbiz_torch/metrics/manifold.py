# coding:utf-8
# @Author: Haowen Wang

import torch
import sklearn.manifold as smn


class tsne(object):
    """
        Wraps sklearn.manifold.TSNE to get TSNE out embedding in gbiz_torch.
    """

    def __init__(self, n_components=2, learning_rate='auto', init='random', perplexity=3):
        self.tsne = smn.TSNE(n_components=n_components,
                             learning_rate=learning_rate, init=init, perplexity=perplexity)

    def __call__(self, input_features):
        """
        input_features: embedding_array-like of shape (n_samples, n_components)
                        Stores the embedding vectors.
        """
        res = self.tsne.fit_transform(input_features)
        return res


class sp_emb(object):
    """
        Wraps sklearn.manifold.SpectralEmbedding to get SpectralEmbedding out in gbiz_torch.
        n_components, int, default=2: The dimension of the projected subspace.
        affinity  {‘nearest_neighbors’, ‘rbf’, ‘precomputed’, ‘precomputed_nearest_neighbors’} or callable, default=’nearest_neighbors’
                  How to construct the affinity matrix.
                  ‘nearest_neighbors’ : construct the affinity matrix by computing a graph of nearest neighbors.

                  ‘rbf’ : construct the affinity matrix by computing a radial basis function (RBF) kernel.

                  ‘precomputed’ : interpret X as a precomputed affinity matrix.

                  ‘precomputed_nearest_neighbors’ : interpret X as a sparse graph of precomputed nearest neighbors, and constructs the affinity matrix by selecting the n_neighbors nearest neighbors.

                  callable : use passed in function as affinity the function takes in data matrix (n_samples, n_features) and return affinity matrix (n_samples, n_samples).
    """

    def __init__(self, n_components=2, affinity='nearest_neighbors'):
        self.sp_emb = smn.SpectralEmbedding(
            n_components=n_components, affinity=affinity)

    def __call__(self, input_features):
        """
        input_features: embedding_array-like of shape (n_samples, n_components)
                        Stores the embedding vectors.
        """
        res = self.sp_emb.fit_transform(input_features)
        return res
