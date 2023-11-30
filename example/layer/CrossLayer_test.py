#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch 
import torch.nn as nn
from gbiz_torch.layer import CrossLayer


# In[3]:


test_input = torch.randn(4, 5)
# padding_idx = 0
# embedding = nn.Embedding(10, 3, padding_idx=padding_idx)
# seq_emb_input = embedding(test_input)


# In[4]:


print(f"test_input.shape is {test_input.shape}")
Cross_layer = CrossLayer(in_shape=test_input.shape[-1], n_layers=3)

test_output = Cross_layer(test_input)

print(f"test_output.shape is {test_output.shape}")
print(f"test_output is {test_output}")


# In[5]:


Cross_layer


# In[ ]:




