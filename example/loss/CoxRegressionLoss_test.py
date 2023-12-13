#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import gbiz_torch.loss import CoxRegressionLoss


# In[ ]:


labels = torch.randint(0, 2, (8, 1))
predict = torch.randint(0, 100, (8, 1))/100

cox_loss = CoxRegressionLoss()
cox_loss(labels, predict)


# In[ ]:




