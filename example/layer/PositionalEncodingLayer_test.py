#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch 
import torch.nn as nn
from gbiz_torch.layer import PositionalEncodingLayer


# In[4]:


test_input = torch.randint(0, 10, (8, 5))
padding_idx = 0
embedding = nn.Embedding(10, 4, padding_idx=padding_idx)
seq_emb_input = embedding(test_input)
seq_emb_input.shape


# In[5]:


# print(f"test_input.shape is {test_input.shape}")
PE_layer = PositionalEncodingLayer(input_length=5, in_shape=4)
test_output = PE_layer(seq_emb_input)

print(f"test_output.shape is {test_output.shape}")
print(f"test_output is {test_output}")


# In[7]:


print(f"test_input.shape is {test_input.shape}")
get_PE_layer = PositionalEncodingLayer(input_length=5, in_shape=4, add_pos=False)
test_output = get_PE_layer(seq_emb_input)

print(f"test_output.shape is {test_output.shape}")
print(f"test_output is {test_output}")


# In[ ]:




