#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch 
import torch.nn as nn
from gbiz_torch.model import FLENModel


# In[3]:


test_input_1 = torch.randint(0, 2, (8, 3))
test_input_2 = torch.randint(0, 3, (8, 4))
test_input_3 = torch.randint(0, 5, (8, 5))
# padding_idx = 0
embedding = nn.Embedding(10, 6)
emb_input1 = embedding(test_input_1)
emb_input2 = embedding(test_input_2)
emb_input3 = embedding(test_input_3)
test_input = [emb_input1, emb_input2, emb_input3]


# ### without context input

# In[4]:


#  = len(test_input)
field_num = len(test_input)
bz, _, dim = test_input[0].shape
hidden_units_list=[16]

dnn_in_shape = torch.cat(test_input, dim=1).reshape(bz, -1).shape[-1]

projection_in_shape = hidden_units_list[-1]+dim

# print(f"field_num is {field_num}, bz is {bz}, dim is {dim}")

FLEN_Model = FLENModel(fields=field_num, in_shape=dim, dnn_in_shape=dnn_in_shape, hidden_units=hidden_units_list, projection_in_shape=projection_in_shape)
test_output = FLEN_Model(test_input)

print(f"test_output.shape is {test_output.shape}")
print(f"test_output is {test_output}")


# In[5]:


FLEN_Model

