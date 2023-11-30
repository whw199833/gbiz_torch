#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch 
import torch.nn as nn
from gbiz_torch.layer import FieldWiseBiInterationLayer


# In[3]:


test_input_1 = torch.randint(0, 2, (8, 3))
test_input_2 = torch.randint(0, 3, (8, 5))
test_input_3 = torch.randint(0, 4, (8, 4))
test_input_4 = torch.randint(0, 5, (8, 4))
# padding_idx = 0
embedding = nn.Embedding(10, 6)
emb_input1 = embedding(test_input_1)
emb_input2 = embedding(test_input_2)
emb_input3 = embedding(test_input_3)
emb_input4 = embedding(test_input_4)
test_input = [emb_input1, emb_input2, emb_input3, emb_input4]


# In[4]:


# print(f"test_input.shape is {test_input.shape}")
## n_inputs: n_fields
## in_shape: emb_dim
n_inputs = len(test_input)
in_shape = test_input[0].shape[-1]

FWBI_layer = FieldWiseBiInterationLayer(n_inputs=n_inputs, in_shape=in_shape, use_bias=True)
test_output = FWBI_layer(test_input)

print(f"test_output.shape is {test_output.shape}")
print(f"test_output is {test_output}")


# In[ ]:




