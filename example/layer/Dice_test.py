#!/usr/bin/env python
# coding: utf-8

# In[8]:


import torch 
from gbiz_torch.layer import Dice


# In[9]:


input = torch.randn((16, 5))
print(f"test_input is {test_input}")
dice_layer = Dice(
    in_shape=test_input.shape, input_length=test_input.shape[1])
test_output = dice_layer(test_input)
print(test_output.shape)
print(f"test_output is {test_output}")


# In[ ]:




