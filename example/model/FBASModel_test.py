#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch 
from gbiz_torch.model import FBASModel


# In[3]:


user_feature = torch.randn(8, 5)
seq_input = torch.randn(8, 3, 4)
item_feature = torch.randn(8, 5)

test_input = [user_feature, seq_input, item_feature]


# ### without extra input

# In[4]:


(bz, dim1) = user_feature.shape 
(bz, seq_length, dim2) = seq_input.shape 
(bz, dim3) = item_feature.shape 


FBAS_Model = FBASModel(in_shape_list=[dim1, dim2, dim3], projection_in_shape=dim1+dim2+dim3)

test_output = FBAS_Model(test_input)

print(test_output.shape)
print(f"test_output is {test_output}")


# ### with extra input

# In[7]:


extra_input = torch.randn(8,5)
(bz, ex_dim) = extra_input.shape
(bz, dim1) = user_feature.shape 
(bz, seq_length, dim2) = seq_input.shape 
(bz, dim3) = item_feature.shape 

projection_in_shape = dim1+dim2+dim3+ex_dim


FBAS_Model = FBASModel(in_shape_list=[dim1, dim2, dim3], projection_in_shape=projection_in_shape)

test_output = FBAS_Model(test_input, extra_input)

print(test_output.shape)
print(f"test_output is {test_output}")


# In[8]:


FBAS_Model


# In[ ]:




