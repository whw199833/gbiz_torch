#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import gbiz_torch.loss import MCCrossEntropy


# In[ ]:


##test MCCrossEntropy

labels_d = torch.cat([torch.eye(5), torch.eye(5)], dim=0)
predict = torch.randn((10, 5), requires_grad=True)

MCCross_Entropy = MCCrossEntropy()
MCCross_Entropy(labels_d, predict)


# In[ ]:




