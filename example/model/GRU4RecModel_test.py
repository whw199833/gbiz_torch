#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch 
import torch.nn as nn
import torch.nn.functional as F
from gbiz_torch.model import GRU4RecModel


# In[3]:


test_input = torch.randint(0, 4, (8, 3))
# padding_idx = 0
embedding = nn.Embedding(4, 6)
test_input = embedding(test_input)
test_input.shape


# ### without context emb

# In[4]:


# print(f"test_input.shape is {test_input.shape}")
## n_inputs: n_fields
## in_shape: emb_dim
(bz, length, dim)= test_input.shape
rnn_out = dim//2
projection_in_shape = rnn_out*length

GRU4Rec_Model = GRU4RecModel(in_shape=dim, rnn_unit=rnn_out, projection_in_shape=projection_in_shape)
test_output = GRU4Rec_Model(test_input)

print(f"test_output.shape is {test_output.shape}")
print(f"test_output is {test_output}")


# ### with context emb

# In[5]:


# print(f"test_input.shape is {test_input.shape}")
## n_inputs: n_fields
## in_shape: emb_dim

context_input = torch.randn(8, 5)
(bz, length, dim) = test_input.shape
(bz_c, dim_c) = context_input.shape

rnn_out = dim//2
projection_in_shape = rnn_out*length+dim_c


GRU4Rec_Model = GRU4RecModel(in_shape=dim, rnn_unit=rnn_out, projection_in_shape=projection_in_shape)
test_output = GRU4Rec_Model(inputs=test_input, extra_input=context_input)

print(f"test_output.shape is {test_output.shape}")
print(f"test_output is {test_output}")


# In[ ]:




