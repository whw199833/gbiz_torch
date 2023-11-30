#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
from gbiz_torch.layer import BiLinearInteractionLayer


# In[3]:


input = torch.randint(0, 3, (2, 3))
get_emb = nn.Embedding(3, 4)
input_seq = get_emb(input)


# In[4]:


test_input = input_seq
BI_layer = BiLinearInteractionLayer(in_shape=test_input.shape[-1])

test_output = BI_layer(test_input)

print(test_output.shape)
print(f"test_output is {test_output}")


# In[5]:


BI_layer
