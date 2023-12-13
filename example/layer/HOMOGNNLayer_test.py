#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges
from gbiz_torch.layer import HOMOGNNLayer


# In[2]:


## test gnn based model or layer

node_data = torch.randn(128, 10)
edge_data = torch.randint(0, 128, (2, 1000))

learn_data = Data(x=node_data, edge_index=edge_data)
learn_data = train_test_split_edges(learn_data)


# In[ ]:


in_channel = learn_data.x.shape[-1]
test_gnn = HOMOGNNLayer(in_channel=in_channel, gtype='ARMAConv', heads=1, hidden_channel_list=[2*in_channel, in_channel//2])

test_inputs = [learn_data.x, learn_data.train_pos_edge_index]
test_gnn(test_inputs).shape

