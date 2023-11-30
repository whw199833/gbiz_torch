#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch 
import torch.nn as nn
from gbiz_torch.model import ContextNetModel


# In[4]:


test_input1 = torch.randn((8, 1, 3))
test_input2 = torch.randn((8, 1, 3))
test_input3 = torch.randn((8, 1, 3))
test_input = torch.cat([test_input1, test_input2, test_input3], dim=1)


# ### without extra input

# In[5]:


# print(f"test_input is {test_input}")
(bz, field_num, dim) = test_input.shape
projection_in_shape=field_num*dim
fields=field_num
in_shape=dim

ContextNet_Model = ContextNetModel(fields=fields, in_shape=in_shape, projection_in_shape=projection_in_shape)
test_output = ContextNet_Model(test_input)

print(test_output.shape)
# print(f"test_output is {test_output}")


# ### with extra input

# In[6]:


# print(f"test_input is {test_input}")
extra_input = torch.randn(8, 2)
(bz, ex_dim) = extra_input.shape

(bz, field_num, dim) = test_input.shape
projection_in_shape=field_num*dim+ex_dim
fields=field_num
in_shape=dim

ContextNet_Model = ContextNetModel(fields=fields, in_shape=in_shape, projection_in_shape=projection_in_shape)
test_output = ContextNet_Model(test_input,extra_input)

print(test_output.shape)
# print(f"test_output is {test_output}")


# In[7]:


ContextNet_Model


# In[ ]:




