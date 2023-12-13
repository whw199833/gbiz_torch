#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import gbiz_torch.loss import LogLoss


# In[ ]:


##test LogLoss

labels = torch.randint(0, 5, (8, 1))
predict = torch.randn((8, 5), requires_grad=True)

Log_loss = LogLoss()
Log_loss(labels, predict)


# In[ ]:




