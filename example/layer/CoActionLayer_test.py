#!/usr/bin/env python
# coding: utf-8

# In[11]:


import torch 
import torch.nn as nn
from gbiz_torch.layer import CoActionLayer


# In[12]:


input_a = torch.randint(0, 3, (8, 10))
get_emb = nn.Embedding(3, 5)
input_seq = get_emb(input_a)
input_item = torch.randn(8, 85)


# In[13]:


test_input = (input_seq, input_item)
# print(f"test_input is {test_input}")
CoA_layer = CoActionLayer(in_shape_list=[test_input[0].shape[-1], test_input[1].shape[-1]])
test_output = CoA_layer(test_input)

print(test_output.shape)
print(f"test_output is {test_output}")


# In[14]:


CoActionLayer


# In[ ]:




