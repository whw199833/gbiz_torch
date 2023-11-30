#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch 
from gbiz_torch.model import TowerModel


# In[5]:


user_input = torch.randn((8, 2))
item_input = torch.randn((8, 3))
test_input = [user_input, item_input]

in_shape_list = [user_input.shape[-1], item_input.shape[-1]]

Tower_Model = TowerModel(in_shape_list=in_shape_list, user_hidden_units=[10, 8, 2], item_hidden_units=[5, 2])

test_output = Tower_Model(test_input)

print(test_output.shape)
print(f"test_output is {test_output}")


# In[6]:


Tower_Model


# In[ ]:




