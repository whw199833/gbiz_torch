#!/usr/bin/env python
# coding: utf-8

# In[7]:


import torch 
from gbiz_torch.model import WndModel


# In[5]:


deep_input = torch.randn((8, 5))
wide_input = torch.randn((8, 10))
print(f"deep_input is {deep_input}")
dnn_layer = WndModel(in_shape=deep_input.shape[-1], hidden_units=[10, 8, 2], use_bn=True, l2_reg=0.9)

test_output = dnn_layer(deep_input, extra_input=wide_input)

print(test_output.shape)
print(f"test_output is {test_output}")


# In[12]:


dnn_layer


# In[ ]:




