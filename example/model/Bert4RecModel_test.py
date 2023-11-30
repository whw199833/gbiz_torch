#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from gbiz_torch.model import Bert4RecModel


# In[3]:


test_input = torch.randint(0, 2, (8, 3))
# padding_idx = 0
embedding = nn.Embedding(2, 6)
test_input = embedding(test_input)
test_input.shape


# ### without context emb

# In[4]:


# print(f"test_input.shape is {test_input.shape}")
# n_inputs: n_fields
# in_shape: emb_dim
(bz, length, dim) = test_input.shape

Bert4Rec_Model = Bert4RecModel(
    in_shape=dim, hidden_units=dim, heads=2, input_length=3, projection_in_shape=length*dim)
test_output = Bert4Rec_Model(test_input)

print(f"test_output.shape is {test_output.shape}")
print(f"test_output is {test_output}")


# ### with context emb

# In[5]:


# print(f"test_input.shape is {test_input.shape}")
# n_inputs: n_fields
# in_shape: emb_dim

context_input = torch.randn(8, 5)
(bz, length, dim) = test_input.shape
(bz_c, dim_c) = context_input.shape


Bert4Rec_Model = Bert4RecModel(in_shape=dim, hidden_units=dim,
                               heads=2, input_length=3, projection_in_shape=length*dim+dim_c)
test_output = Bert4Rec_Model(inputs=test_input, extra_input=context_input)

print(f"test_output.shape is {test_output.shape}")
print(f"test_output is {test_output}")


# In[ ]:
