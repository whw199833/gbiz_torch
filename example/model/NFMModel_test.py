#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch 
import torch.nn as nn
from gbiz_torch.model import NFMModel


# In[3]:


test_1 = torch.randn(8, 1, 5)
test_2 = torch.randn(8, 1, 5)
test_3 = torch.randn(8, 1, 5)

test_input = torch.cat([test_1, test_2, test_3], dim=1)
test_input.shape


# ### keep_fm_dim

# In[7]:


print(f"test_input.shape is {test_input.shape}")
NFM_Model = NFMModel(keep_fm_dim=True, in_shape=test_input.shape[-1], hidden_units=[10, 2])

test_output = NFM_Model(test_input)

print(f"test_output.shape is {test_output.shape}")
print(f"test_output is {test_output}")


# ### do not keep fm dim

# In[8]:


print(f"test_input.shape is {test_input.shape}")
NFM_Model = NFMModel(keep_fm_dim=False, in_shape=test_input.shape[-1], hidden_units=[10, 2])

test_output = NFM_Model(test_input)

print(f"test_output.shape is {test_output.shape}")
print(f"test_output is {test_output}")


# In[9]:


NFM_Model


# In[ ]:




