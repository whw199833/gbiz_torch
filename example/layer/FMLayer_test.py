#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch 
import torch.nn as nn
from gbiz_torch.layer import FMLayer


# In[3]:


test_input = torch.randint(0, 10, (4, 5))
padding_idx = 0
embedding = nn.Embedding(10, 3, padding_idx=padding_idx)
seq_emb_input = embedding(test_input)


# In[6]:


print(f"seq_emb_input.shape is {seq_emb_input.shape}")
FM_layer = FMLayer(keep_dim=True)

test_output = FM_layer(seq_emb_input)

print(f"test_output.shape is {test_output.shape}")
print(f"test_output is {test_output}")


# In[7]:


FM_layer


# In[ ]:




