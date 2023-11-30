#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch 
import torch.nn as nn
import torch.nn.functional as F
from gbiz_torch.layer import MultiHeadAttentionLayer


# In[3]:


test_input = torch.randint(0, 2, (8, 3))
# padding_idx = 0
embedding = nn.Embedding(2, 6)
test_input = embedding(test_input)
test_input.shape


# ### without mask

# In[4]:


# print(f"test_input.shape is {test_input.shape}")
## n_inputs: n_fields
## in_shape: emb_dim

MHA_layer = MultiHeadAttentionLayer(in_shape=6, hidden_units=4, heads=2)
test_output = MHA_layer(test_input)

print(f"test_output.shape is {test_output.shape}")
print(f"test_output is {test_output}")


# ### with mask

# In[6]:


(bz, seq, dim)= test_input.shape
in_shape = dim
hidden_units = 4
heads = 4
mask = torch.randint(0, 2, (bz, heads, seq, seq))

MHA_layer = MultiHeadAttentionLayer(in_shape=in_shape, hidden_units=hidden_units, heads=heads)
test_output = MHA_layer(test_input, mask=mask)

print(f"test_output.shape is {test_output.shape}")
print(f"test_output is {test_output}")


# In[ ]:




