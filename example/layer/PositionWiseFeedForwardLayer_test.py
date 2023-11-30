#!/usr/bin/env python
# coding: utf-8

# In[4]:


import torch 
import torch.nn as nn
import torch.nn.functional as F
from gbiz_torch.layer import PositionWiseFeedForwardLayer


# In[5]:


test_input = torch.randint(0, 2, (8, 3))
# padding_idx = 0
embedding = nn.Embedding(2, 6)
test_input = embedding(test_input)
test_input.shape


# In[7]:


# print(f"test_input.shape is {test_input.shape}")
## n_inputs: n_fields
## in_shape: emb_dim

PWF_layer = PositionWiseFeedForwardLayer(in_shape=6, intermediate_size=12, hidden_units=2)
test_output = PWF_layer(test_input)

print(f"test_output.shape is {test_output.shape}")
print(f"test_output is {test_output}")


# In[ ]:




