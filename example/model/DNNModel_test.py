#!/usr/bin/env python
# coding: utf-8

# In[6]:


import torch 
from gbiz_torch.model import DNNModel


# In[7]:


deep_input = torch.randn(8, 10)
wide_input = torch.randn(8, 5)

# test_input = [user_feature, seq_input, item_feature]


# In[9]:


dnn_layer = DNNModel(in_shape=deep_input.shape[-1], hidden_units=[10, 8, 2], use_bn=True, l2_reg=0.9)

test_output = dnn_layer(deep_input)

print(test_output.shape)
print(f"test_output is {test_output}")


# In[10]:


dnn_layer


# In[ ]:




