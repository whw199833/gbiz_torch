#!/usr/bin/env python
# coding: utf-8

# In[8]:


import torch 
import torch.nn as nn
import numpy as np
from gbiz_torch.model import DCAPModel


# ### without extra input

# In[9]:


test_input = torch.randint(0, 3, (8, 5))
get_emb = nn.Embedding(3, 16)
test_input = get_emb(test_input)
(bz, seq_length, dim) = test_input.shape


# In[10]:


fields = 4
projection_in_shape = sum(np.arange(fields))

DCAP_Model = DCAPModel(fields=fields, in_shape=dim, projection_in_shape=projection_in_shape)

test_output = DCAP_Model(test_input)

print(test_output.shape)
print(f"test_output is {test_output}")


# ### with extra input

# In[14]:


fields = 4
extra_input = torch.randn(8,2)

(bz, ex_dim) = extra_input.shape
projection_in_shape = sum(np.arange(fields))+ex_dim


DCAP_Model = DCAPModel(fields=fields, in_shape=dim, projection_in_shape=projection_in_shape)

test_output = DCAP_Model(test_input, extra_input=extra_input)

print(test_output.shape)
print(f"test_output is {test_output}")


# In[15]:


DCAP_Model


# In[ ]:




