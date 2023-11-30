#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch 
import torch.nn as nn
from gbiz_torch.layer import ContextNetBlockLayer


# In[3]:


input_a = torch.randint(0, 3, (8, 10))
get_emb = nn.Embedding(3, 5)
input_seq = get_emb(input_a)


# In[4]:


# print(f"test_input is {test_input}")
CNB_layer = ContextNetBlockLayer(fields=input_seq.shape[1], in_shape=input_seq.shape[2])
test_output = CNB_layer(input_seq)

print(test_output.shape)
# print(f"test_output is {test_output}")


# In[5]:


ContextNetBlockLayer


# In[ ]:




