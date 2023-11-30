#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch 
import torch.nn as nn
from gbiz_torch.layer import WeightedSeqAggLayer


# ### 单seq输入：

# In[4]:


input_a = torch.randint(0, 3, (8, 10))
get_emb = nn.Embedding(3, 5)
input_seq = get_emb(input_a)

# print(f"test_input is {test_input}")
WSA_layer = WeightedSeqAggLayer(in_shape_list=[input_seq.shape[-1]], n_inputs=1)
test_output = WSA_layer(input_seq)

print(test_output.shape)
# print(f"test_output is {test_output}")


# ### 双输入：

# In[12]:


# print(f"test_input is {test_input}")
input_a = torch.randint(0, 3, (8, 10))
get_emb = nn.Embedding(3, 5)
input_seq = get_emb(input_a)

input_item_emb = torch.randn(8, 3)
test_input = (input_seq, input_item_emb)

# mask_input=None
mask_input = torch.randint(0, 2, (8, 10))

WSA_layer = WeightedSeqAggLayer(in_shape_list=[test_input[0].shape[-1], test_input[1].shape[-1]], n_inputs=2)
test_output = WSA_layer(test_input, mask=mask_input)

print(test_output.shape)
# print(f"test_output is {test_output}")


# In[7]:


WeightedSeqAggLayer


# In[ ]:




