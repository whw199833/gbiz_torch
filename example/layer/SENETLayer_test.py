#!/usr/bin/env python
# coding: utf-8

# In[32]:


import torch 
from gbiz_torch.layer import SENETLayer


# In[33]:


test_input1 = torch.unsqueeze(torch.randn((8, 5)), 1)
test_input2 = torch.unsqueeze(torch.randn((8, 5)), 1)
test_input = torch.cat([test_input1, test_input2], dim=1)
test_input.shape


# In[34]:


torch.mean(test_input, dim=-1, keepdim=True).shape


# In[35]:


print(f"test_input.shape is {test_input.shape}")
sene_layer = SENETLayer(fields=2, reduction_ratio=1)
test_output = sene_layer(test_input)

print(f"test_output.shape is {test_output.shape}")
print(f"test_output is {test_output}")


# In[36]:


sene_layer


# In[ ]:




