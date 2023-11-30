#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch 
import torch.nn as nn
from gbiz_torch.layer import FineSeqLayer


# In[4]:


test_input = torch.randint(0, 3, (8, 3))
# padding_idx = 0
embedding = nn.Embedding(3, 5)
seq_emb_input = embedding(test_input)

item_input = torch.randn(8, 10)
test_input = [seq_emb_input, item_input]


# In[5]:


# print(f"test_input.shape is {test_input.shape}")

in_shape_list = [seq_emb_input.shape[-1], item_input.shape[-1]]

FS_layer = FineSeqLayer(in_shape_list=in_shape_list)

test_output = FS_layer(test_input)

print(f"test_output.shape is {test_output.shape}")
print(f"test_output is {test_output}")


# In[5]:


Cross_layer


# In[ ]:




