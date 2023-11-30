#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch 
from gbiz_torch.layer import CGCGatingNetworkLayer


# In[5]:


expert1_output = torch.unsqueeze(torch.randn(8, 10), dim=1)
expert2_output = torch.unsqueeze(torch.randn(8, 10), dim=1)
expert3_output = torch.unsqueeze(torch.randn(8, 10), dim=1)

expert4_output = torch.unsqueeze(torch.randn(8, 10), dim=1)
expert5_output = torch.unsqueeze(torch.randn(8, 10), dim=1)

task_expert_input = torch.cat((expert1_output, expert2_output, expert3_output), dim=1)
shared_expert_input = torch.cat((expert4_output, expert5_output), dim=1)
input = torch.mean(torch.cat((expert1_output, expert2_output, expert3_output, expert4_output, expert5_output), dim=1), dim=1)

task_expert_input.shape, shared_expert_input.shape, input.shape


# In[7]:


test_input = (task_expert_input, shared_expert_input, input)
CGCGN_layer = CGCGatingNetworkLayer(in_shape=test_input[-1].shape[-1], total_experts=5)

test_output = CGCGN_layer(test_input)

print(test_output.shape)
print(f"test_output is {test_output}")


# In[8]:


CGCGatingNetworkLayer

