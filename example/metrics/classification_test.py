#!/usr/bin/env python
# coding: utf-8

# In[2]:


from gbiz_torch.metrics import AUC, Confusion_Matrix, ACC_F1_score, Top_K_Acc, Multi_Class_RP


# In[3]:


## test AUC

import torch
# simulate a classification problem
preds = torch.randn(10, 1)
# target = torch.randn(10, 1)
target = torch.randint(0,2, (10,))

eval_auc = AUC()
eval_auc(target, preds)


# In[ ]:


## test Confusion_Matrix

import torch
# simulate a classification problem
preds = torch.randn(10, 3)
pred_label = torch.argmax(preds, axis=1)

# target = torch.randn(10, 1)
target = torch.randint(0,3, (10,))


C_matrix = Confusion_Matrix()
C_matrix(pred_label, target)


# In[ ]:


## test F1

import torch
# simulate a classification problem
preds = torch.randn(100, 4)
pred_label = torch.argmax(preds, axis=1)

# target = torch.randn(10, 1)
target = torch.randint(0,4, (100,))


f1_acc = ACC_F1_score()
f1_acc(pred_label, target)


# In[ ]:


## test Top_K_Acc

import torch
# simulate a classification problem
preds = torch.randn(100, 4)

target = torch.randint(0,4, (100,))


top_k_acc = Top_K_Acc(k=2)
top_k_acc(target, preds)


# In[ ]:


## test Multi_Class_RP

import torch
# simulate a classification problem
preds = torch.randn(10, 4)
preds_labels = torch.argmax(preds, axis=-1)

target = torch.randint(0,4, (10,))


mc_rp = Multi_Class_RP()
precision, recall, fbeta_score, support = mc_rp(target, preds_labels)
print(f"precision is {precision}")
print(f"recall is {recall}")
print(f"fbeta_score is {fbeta_score}")
print(f"support is {support}")

