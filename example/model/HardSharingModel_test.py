#!/usr/bin/env python
# coding: utf-8

# In[4]:


import torch 
from gbiz_torch.model import HardSharingModel


# In[7]:


test_input = torch.randn((8, 5))
# print(f"test_input is {test_input}")
HDS_layer = HardSharingModel(in_shape=test_input.shape[-1], sharing_hidden_units=[100, 50], task_hidden_units=[[10,1], [5,2], [20,1]])

test_output = HDS_layer(test_input)

# print(test_output.shape)
print(f"test_output is {test_output}")


# In[8]:


HardSharingModel


# In[ ]:




