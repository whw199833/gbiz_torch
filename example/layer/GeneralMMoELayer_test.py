#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch 
import torch.nn as nn
from gbiz_torch.layer import GeneralMMoELayer


# In[7]:


shared_inputs = torch.randn(4, 10)
expert_inputs = torch.randn(4, 5, 20)

test_input = [shared_inputs, expert_inputs]


# In[9]:


gmmoe_layer = GeneralMMoELayer(
    shared_in_shape=test_input[0].shape[-1], 
    expert_in_shape=test_input[1].shape[-1],
    num_experts=test_input[1].shape[-1],
    num_tasks=3
    )

test_output = gmmoe_layer(test_input)

# print(f"test_output.shape is {test_output.shape}")
print(f"test_output is {test_output}")


# In[10]:


GeneralMMoELayer


# In[14]:


combined_out = torch.stack(test_output)
combined_out.shape


# In[16]:


combined_out = combined_out.permute((1, 0, 2))
combined_out.shape


# In[ ]:




