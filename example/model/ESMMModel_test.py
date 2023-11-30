#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch 
from gbiz_torch.model import ESMMModel


# In[5]:


test_input = torch.randn((8, 5))
(bz, dim) = test_input.shape
ESMM_Model = ESMMModel(in_shape=dim, task_hidden_units=[[10, 1], [4, 1]])

test_output = ESMM_Model(inputs=test_input)

# print(test_output.shape)
print(f"test_output is {test_output}")


# In[6]:


ESMM_Model


# In[ ]:




