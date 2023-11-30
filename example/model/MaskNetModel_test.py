#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch 
import torch.nn as nn
from gbiz_torch.model import MaskNetModel


# In[3]:


input_a = torch.randint(0, 3, (8, 10))
get_emb = nn.Embedding(3, 5)
input_seq = get_emb(input_a)
input_item = torch.randn(8, 6)
# avg_past_item = torch.randn(8, 8)


# In[4]:


test_input = torch.randn(8, 3)

print(test_input.shape)
# print(f"test_output is {test_output}")


# ### without extra input

# In[6]:


(bz, dim) = test_input.shape
projection_in_shape=dim
projection_hidden_units=[4, 2]

MaskNet_Model = MaskNetModel(in_shape=dim, projection_in_shape=projection_in_shape, projection_hidden_units=projection_hidden_units)
test_output = MaskNet_Model(test_input)

print(test_output.shape)


# ### with extra input

# In[9]:


extra_input = torch.randn(8, 2)

(bz, ex_dim) = extra_input.shape
(bz, dim) = test_input.shape

projection_in_shape=dim+ex_dim
projection_hidden_units=[4, 2]


MaskNet_Model = MaskNetModel(in_shape=dim, projection_in_shape=projection_in_shape, projection_hidden_units=projection_hidden_units)
test_output = MaskNet_Model(test_input, extra_input=extra_input)

print(test_output.shape)


# In[10]:


MaskNet_Model


# In[ ]:




