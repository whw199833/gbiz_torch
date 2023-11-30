#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch 
from gbiz_torch.model import CANModel


# In[3]:


seq_input = torch.randn((8, 3, 5))
item_input = torch.randn((8, 24))

test_input = (seq_input, item_input)


# ### without context input

# In[4]:


(bz, seq_lenght, dim_a) = seq_input.shape
(bz, dim_b) = item_input.shape

in_shape_list=[dim_a, dim_b]
hidden_units=[4]
projection_in_shape=hidden_units[-1]

CAN_Model = CANModel(in_shape_list=in_shape_list, hidden_units=hidden_units, projection_in_shape=projection_in_shape)

test_output = CAN_Model(test_input)

print(test_output.shape)
print(f"test_output is {test_output}")


# ### with context input

# In[5]:


a = torch.randn(8,3,4)
b = torch.randn(8,3,2)
torch.cat([a,b], dim=-1)


# In[6]:


extra_input = torch.randn(8,2)
(bz, ex_dim) = extra_input.shape
(bz, seq_lenght, dim_a) = seq_input.shape
(bz, dim_b) = item_input.shape

in_shape_list=[dim_a, dim_b]
hidden_units=[4]
projection_in_shape=hidden_units[-1]+ex_dim

CAN_Model = CANModel(in_shape_list=in_shape_list, hidden_units=hidden_units, projection_in_shape=projection_in_shape)

test_output = CAN_Model(test_input, extra_input)

print(test_output.shape)
print(f"test_output is {test_output}")


# In[ ]:




