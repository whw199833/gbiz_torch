#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch 
from gbiz_torch.model import GeneralMMoEModel


# In[6]:


test_input = torch.randn((8, 5))
print(f"test_input.shape is {test_input.shape}")

(bz, dim) = test_input.shape
GMMOE_Model = GeneralMMoEModel(in_shape=dim, sharing_hidden_units=[10, 8], experts_hidden_units=[[2], [4, 2], [8, 2]], task_hidden_units=[[1], [2, 1], [1]])

test_output = GMMOE_Model(test_input)

# print(test_output.shape)
print(f"test_output is {test_output}")


# In[7]:


GMMOE_Model


# In[ ]:




