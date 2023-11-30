#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch 
import torch.nn as nn
import torch.nn.functional as F
from gbiz_torch.model import FiBiNetModel


# In[3]:


test_input1 = torch.randn(8, 1, 5)
test_input2 = torch.randn(8, 1, 5)
test_input3 = torch.randn(8, 1, 5)

test_input = torch.cat([test_input1, test_input2, test_input3], dim=1)
test_input.shape


# ### without extra input

# In[4]:


# print(f"test_input.shape is {test_input.shape}")
## n_inputs: n_fields
## in_shape: emb_dim
(bz, fields_num, dim) = test_input.shape

projection_in_shape = dim+1

FiBiNet_Model = FiBiNetModel(fields=fields_num, in_shape=dim, reduction_ratio=2, projection_in_shape=projection_in_shape)
test_output = FiBiNet_Model(test_input)

print(f"test_output.shape is {test_output.shape}")
print(f"test_output is {test_output}")


# ### plus context input

# In[5]:


# print(f"test_input.shape is {test_input.shape}")
## n_inputs: n_fields
## in_shape: emb_dim
context_input = torch.randn(8, 3)
(bz, ex_dim) = context_input.shape
(bz, fields_num, dim) = test_input.shape

projection_in_shape = dim+1+ex_dim

FiBiNet_Model = FiBiNetModel(fields=fields_num, in_shape=dim, reduction_ratio=2, projection_in_shape=projection_in_shape)
test_output = FiBiNet_Model(inputs=test_input, extra_input=context_input)

print(f"test_output.shape is {test_output.shape}")
print(f"test_output is {test_output}")


# In[6]:


FiBiNet_Model


# In[ ]:




