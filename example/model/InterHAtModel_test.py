#!/usr/bin/env python
# coding: utf-8

# In[4]:


import torch
from gbiz_torch.model import InterHAtModel


# In[5]:


test_input = torch.randn(8, 3, 10)


# In[6]:


(bz, fields, dim) = test_input.shape

in_shape = dim
hidden_units = dim
projection_in_shape = dim

InterHAt_Model = InterHAtModel(in_shape=in_shape, hidden_units=hidden_units, projection_in_shape=projection_in_shape, heads=5)

test_output = InterHAt_Model(test_input)

print(test_output.shape)
print(f"test_output is {test_output}")


# In[7]:


InterHAt_Model


# In[ ]:




