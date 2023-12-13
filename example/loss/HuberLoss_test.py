#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import gbiz_torch.loss import HuberLoss


# In[ ]:


##test HuberLoss

labels = torch.randn(8, 5)
predict = torch.randn(8, 5)

Huber_Loss = HuberLoss()
Huber_Loss(labels, predict)


# In[ ]:




