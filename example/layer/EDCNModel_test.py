#!/usr/bin/env python
# coding: utf-8

# In[6]:


import torch 
from gbiz_torch.model import EDCNModel


# In[7]:


test_input = torch.randn((8, 5))
# print(f"test_input is {test_input}")
EDCN_Model = EDCNModel(in_shape=test_input.shape[-1], dropout_rate=0.3, use_bn=True, l2_reg=0.9)

test_output = EDCN_Model(test_input)

print(test_output.shape)
print(f"test_output is {test_output}")


# In[8]:


EDCN_Model


# In[ ]:




