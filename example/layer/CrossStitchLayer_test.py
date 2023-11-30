#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch 
import torch.nn as nn
from gbiz_torch.layer import CrossStitchLayer


# In[4]:


test_input = []
in_shape_list = []
for i in range(5):
    tmp = torch.randn((4, 5))
    test_input.append(tmp)
    in_shape_list.append(tmp.shape[-1])

print(f"test_input is {test_input}")


# In[5]:


CrossStitch_layer = CrossStitchLayer(in_shape_list=in_shape_list)

test_output = CrossStitch_layer(test_input)

# print(f"test_output.shape is {test_output.shape}")
print(f"test_output is {test_output}")


# In[6]:


CrossStitch_layer


# In[9]:


torch.stack(test_output).shape


# In[ ]:




