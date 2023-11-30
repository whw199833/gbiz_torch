#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch 
import torch.nn as nn
from gbiz_torch.layer import BridgeLayer


# In[12]:


test_input = torch.randn(8, 6)


# In[13]:


BL_layer = BridgeLayer(in_shape=test_input.shape[-1], n_layers=5)

test_output = BL_layer(test_input)

print(f"test_output.shape is {test_output.shape}")
print(f"test_output is {test_output}")


# In[5]:


Cross_layer


# In[ ]:




