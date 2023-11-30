#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch 
import torch.nn as nn
from gbiz_torch.layer import DCAPLayer


# In[3]:


test_input = torch.randint(0, 3, (8, 5))
get_emb = nn.Embedding(3, 16)
seq_input = get_emb(test_input)
seq_input.shape


# ### without mask

# In[12]:


DCAP_layer = DCAPLayer(fields=3, in_shape=seq_input.shape[-1])

test_output = DCAP_layer(seq_input)

print(f"test_output.shape is {test_output.shape}")
print(f"test_output is {test_output}")


# In[5]:


DCAP_layer


# In[ ]:




