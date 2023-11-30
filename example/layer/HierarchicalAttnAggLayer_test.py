#!/usr/bin/env python
# coding: utf-8

# In[46]:


import torch 
import torch.nn as nn
import torch.nn.functional as F
from gbiz_torch.layer import HierarchicalAttnAggLayer


# In[49]:


test_input = torch.randint(0, 2, (8, 3))
# padding_idx = 0
embedding = nn.Embedding(2, 6)
test_input = embedding(test_input)
test_input.shape


# In[50]:


# print(f"test_input.shape is {test_input.shape}")
## n_inputs: n_fields
## in_shape: emb_dim

HAA_layer = HierarchicalAttnAggLayer(in_shape=6, n_layers=3)
test_output = HAA_layer(test_input)

print(f"test_output.shape is {test_output.shape}")
print(f"test_output is {test_output}")


# In[ ]:




