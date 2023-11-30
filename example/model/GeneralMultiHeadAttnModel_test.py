#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch 
import torch.nn as nn
import torch.nn.functional as F
from gbiz_torch.model import GeneralMultiHeadAttnModel


# In[3]:


test_input = torch.randint(0, 2, (8, 3))
# padding_idx = 0
embedding = nn.Embedding(2, 6)
test_input = embedding(test_input)
test_input.shape


# ### without extra input

# In[4]:


# print(f"test_input.shape is {test_input.shape}")
## n_inputs: n_fields
## in_shape: emb_dim
(bz, seq_length, dim) = test_input.shape
hidden_units = 10

GMHA_Model = GeneralMultiHeadAttnModel(in_shape=dim, hidden_units=hidden_units, projection_in_shape=seq_length*hidden_units, heads=5, projection_hidden_units=[8, 1])
test_output = GMHA_Model(test_input)

print(f"test_output.shape is {test_output.shape}")
print(f"test_output is {test_output}")


# ### with extra input

# In[5]:


# print(f"test_input.shape is {test_input.shape}")
## n_inputs: n_fields
## in_shape: emb_dim
context_input = torch.randn(8, 10)
(bz, ex_input_dim) = context_input.shape

(bz, seq_length, dim) = test_input.shape
hidden_units = 10

GMHA_Model = GeneralMultiHeadAttnModel(in_shape=dim, hidden_units=hidden_units, projection_in_shape=seq_length*hidden_units+ex_input_dim, heads=5)
test_output = GMHA_Model(inputs=test_input, extra_input=context_input)

print(f"test_output.shape is {test_output.shape}")
print(f"test_output is {test_output}")


# In[ ]:




