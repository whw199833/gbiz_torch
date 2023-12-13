#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges
from gbiz_torch.layer import HETEGNNLayer


# In[2]:


## test hetegnn based model or layer
from torch_geometric.data import HeteroData
import torch

data = HeteroData()
data['paper'].x = torch.randn(10, 5)
data['author'].x = torch.randn(4, 15)
data['conference'].x = torch.randn(3, 20)

author_t = torch.randint(0, 4, (15,))
paper_t = torch.randint(0, 10, (15,))
data['author', 'paper'].edge_index = torch.stack([author_t, paper_t])
data['paper', 'author'].edge_index = torch.stack([paper_t, author_t])

paper_t = torch.arange(10)
conference_t = torch.randint(0, 2, (10,))

data['paper', 'conference'].edge_index = torch.stack([paper_t, conference_t])
data['paper', 'cites', 'paper'].edge_index = torch.randint(0, 10, (2, 10))


# In[ ]:


test_gnn = HETEGNNLayer(data.node_types, data.metadata(), gtype='HANConv', hidden_channels=10, out_channels=5, num_heads=2, num_layers=3)

test_gnn(data.x_dict, data.edge_index_dict)

