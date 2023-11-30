#!/usr/bin/env python
# coding: utf-8

# In[10]:


import torch 
from gbiz_torch.layer import DNNLayer


# In[11]:


test_input = torch.randn((16, 5))
print(f"test_input is {test_input}")
dnn_layer = DNNLayer(in_shape=test_input.shape[-1], hidden_units=[10, 8, 1], activation='leaky_relu', dropout_rate=0.3, use_bn=True, l2_reg=0.9)

test_output = dnn_layer(test_input)

print(test_output.shape)
print(f"test_output is {test_output}")


# In[12]:


dnn_layer


# In[ ]:




