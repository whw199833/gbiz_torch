#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch 
import torch.nn as nn
from gbiz_torch.model import DCNModel


# In[3]:


test_input = torch.randn(4, 5)
# padding_idx = 0
# embedding = nn.Embedding(10, 3, padding_idx=padding_idx)
# seq_emb_input = embedding(test_input)


# In[4]:


print(f"test_input.shape is {test_input.shape}")

(bz, dim) = test_input.shape
DCN_model = DCNModel(in_shape=dim, hidden_units=[dim//2, 1])

test_output = DCN_model(test_input)

print(f"test_output.shape is {test_output.shape}")
print(f"test_output is {test_output}")


# In[5]:


Cross_layer


# In[ ]:




