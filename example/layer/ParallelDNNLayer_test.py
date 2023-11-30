#!/usr/bin/env python
# coding: utf-8

# In[4]:


import torch 
import torch.nn as nn 
from gbiz_torch.layer import ParallelDNNLayer


# In[11]:


test_input = torch.randn(4, 5)
# get_emb = nn.Embedding(3, 4)
# seq_input = get_emb(input)
print(f"test_input is {test_input.shape}")
PDNN_layer = ParallelDNNLayer(in_shape=test_input.shape[-1], hidden_units=[10, 8, 2], n_experts=3)

test_output = PDNN_layer(test_input)

print(test_output.shape)
print(f"test_output is {test_output}")


# In[12]:


PDNN_layer


# In[ ]:




