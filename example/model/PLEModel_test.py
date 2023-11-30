#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch 
from gbiz_torch.model import PLEModel


# In[2]:


test_input = torch.randn((8, 5))

in_shape = test_input.shape[-1]
expert_units = [20, 15, 10]
n_task_experts = [[1, 1], [1, 1], [1, 1]]
n_shared_experts = [2, 2, 2]
task_hidden_units = [[10, 4, 1], [10, 1]]


PLE_Model = PLEModel(in_shape=in_shape, expert_units=expert_units, n_task_experts=n_task_experts, \
                            n_shared_experts=n_shared_experts, task_hidden_units=task_hidden_units)

test_output = PLE_Model(test_input)

# print(test_output.shape)
print(f"test_output is {test_output}")


# In[3]:


PLE_Model


# In[ ]:




