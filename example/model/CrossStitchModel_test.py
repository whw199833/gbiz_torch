#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch 
import torch.nn as nn
from gbiz_torch.model import CrossStitchModel


# In[3]:


test_input = torch.randn(8, 5)

print(f"test_input is {test_input}")


# In[4]:


(bz, dim) = test_input.shape

CrossStitch_Model = CrossStitchModel(in_shape=dim)

test_output = CrossStitch_Model(test_input)

# print(f"test_output.shape is {test_output.shape}")
print(f"test_output is {test_output}")


# In[5]:


CrossStitch_Model


# In[ ]:




