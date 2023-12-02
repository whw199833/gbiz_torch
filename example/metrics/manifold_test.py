#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
from gbiz_torch.metrics import tsne, sp_emb


# In[3]:


# test tsne

layer_out = torch.randn(10, 4096)

get_tsne = tsne(n_components=3, learning_rate='auto', init='random', perplexity=3)
get_tsne(layer_out)


# In[ ]:


# test sp_emb

layer_out = torch.randn(10, 4096)

get_tsne = tsne(n_components=3, learning_rate='auto', init='random', perplexity=3)
get_tsne(layer_out)

