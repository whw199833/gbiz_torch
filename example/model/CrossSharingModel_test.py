#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch 
import torch.nn as nn
from gbiz_torch.model import CrossSharingModel


# In[4]:


test_input = torch.randn(4, 5)
# padding_idx = 0
# embedding = nn.Embedding(10, 3, padding_idx=padding_idx)
# seq_emb_input = embedding(test_input)


# In[5]:


print(f"test_input.shape is {test_input.shape}")

(bz, dim) = test_input.shape
CSMTL_model = CrossSharingModel(in_shape=dim, dcn_hidden_units=[dim//2], task_hidden_units=[[1], [2]])

test_output = CSMTL_model(test_input)

print(f"test_output.shape is {len(test_output)}")
print(f"test_output is {test_output}")


# In[6]:


CrossSharingModel


# In[ ]:




