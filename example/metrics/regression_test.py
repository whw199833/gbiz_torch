#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
from gbiz_torch.metrics import r2_score, MME


# In[3]:


### test r2_score
y_pred = torch.randn(10, 1)
y_label = torch.randint(0, 2, (10,))

get_r2 = r2_score()
get_r2(y_label, y_pred)


# In[ ]:


### test MME
y_pred = torch.randn(10, 4)
y_label = torch.randn(10, 4)

get_mme = MME()
get_mme(y_label, y_pred)

