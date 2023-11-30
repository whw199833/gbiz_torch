#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch 
import torch.nn as nn
import torch.nn.functional as F
from gbiz_torch.model import DINModel


# In[3]:


test_input1 = torch.randint(0, 2, (8, 3))
# padding_idx = 0
embedding = nn.Embedding(2, 6)
test_input1 = embedding(test_input1)
test_input1.shape


# In[4]:


test_input2 = torch.randint(0, 2, (8, 1))
# padding_idx = 0
embedding = nn.Embedding(2, 6)
test_input2 = torch.squeeze(embedding(test_input2), dim=1)
# test_input2 = torch.squeeze(test_input2, dim=1)
test_input2.shape


# ### seq input + item input

# In[5]:


# print(f"test_input.shape is {test_input.shape}")
## n_inputs: n_fields
## in_shape: emb_dim

test_input = [test_input1, test_input2]

DIN_Model = DINModel(in_shape=test_input[0].shape[-1], input_length=test_input[0].shape[1], projection_in_shape=test_input[1].shape[-1])
test_output = DIN_Model(test_input)

print(f"test_output.shape is {test_output.shape}")
print(f"test_output is {test_output}")


# ### plus context input

# In[6]:


# print(f"test_input.shape is {test_input.shape}")
## n_inputs: n_fields
## in_shape: emb_dim

test_input = [test_input1, test_input2]
context_input = torch.randn(8, 3)

DIN_Model = DINModel(in_shape=test_input[0].shape[-1], input_length=test_input[0].shape[1], projection_in_shape=test_input[1].shape[-1]+context_input.shape[-1])
test_output = DIN_Model(inputs=test_input, extra_input=context_input)

print(f"test_output.shape is {test_output.shape}")
print(f"test_output is {test_output}")


# In[7]:


DIN_Model

