#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch 
import torch.nn as nn
import torch.nn.functional as F
from gbiz_torch.layer import FuseLayer


# In[3]:


test_input1 = torch.randint(0, 2, (8, 3))
# padding_idx = 0
embedding = nn.Embedding(2, 6)
test_input1 = embedding(test_input1)
test_input1.shape


# In[10]:


test_input2 = torch.randint(0, 2, (8, 1))
# padding_idx = 0
embedding = nn.Embedding(2, 6)
test_input2 = embedding(test_input2)
# test_input2 = torch.squeeze(test_input2, dim=1)
test_input2.shape


# In[11]:


# print(f"test_input.shape is {test_input.shape}")
## n_inputs: n_fields
## in_shape: emb_dim

test_input = [test_input1, test_input2]

FS_layer = FuseLayer(in_shape=6, input_length=3, hidden_units=[12])
test_output = FS_layer(test_input)

print(f"test_output.shape is {test_output.shape}")
print(f"test_output is {test_output}")


# In[ ]:




