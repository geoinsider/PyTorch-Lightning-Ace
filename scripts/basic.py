#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/suredream/PyTorch-Lightning-Ace/blob/main/pll-basic.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:


"""
resnet
Instance Segmentation of Agricultural Fields
delineate agricultural field parcels from satellite images via deep learning instance segmentation

fully convolutional instance segmentation architecture

rgb, sentinel-2, 
model unet
dataset 

experiments


## Concepts
Instance Segmentation
Semantic Segmentation

## Reference
- https://github.com/chrieke/InstanceSegmentation_Sentinel2
- https://github.com/waldnerf/decode mxnet no pre-trained weights
- https://rising.readthedocs.io/en/stable/lightning_segmentation.html
- https://colab.research.google.com/github/fepegar/torchio-notebooks/blob/main/notebooks/TorchIO_MONAI_PyTorch_Lightning.ipynb#scrollTo=XcdPOVf7WbyQ
"""

# ![ ! -d "hymenoptera_data" ] && wget https://download.pytorch.org/tutorial/hymenoptera_data.zip && unzip hymenoptera_data.zip

import pytorch_lightning as pl
import pl_bolts

print(f"pl version: {pl.__version__}")
print(f"pl_bolts version: {pl_bolts.__version__}")


# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


import matplotlib.pyplot as plt
import numpy as np
import time
import os
import copy


# In[ ]:


# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=2)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[ ]:


class CustomModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        # https://stackoverflow.com/questions/66000358/how-to-strip-a-pretrained-network-and-add-some-layers-to-it-using-pytorch-lightn
        
    def forward(self,x):
        in_size = x.size(0)
        out = self.conv1(x)
        out = F.relu(out)
        out = F.max_pool2d(out, 2, 2)
        out = self.conv2(out)
        out = F.relu(out)
        out = out.view(in_size, -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.log_softmax(out, dim=1)
        return out
    
    def training_step(self, batch, idx):
        x, y = batch
        _y = self(x)
        # 计算loss
        loss = self.train_criterion(_y, y)
        # 更新结果
        self.train_metric.update(_y, y)
        return loss

    def training_epoch_end(self, outs):
        # 计算平均loss
        loss = 0.
        for out in outs:
            loss += out["loss"].cpu().detach().item()
        loss /= len(outs)
        # 计算指标
        acc, f1 = self.train_metric.compute()
        # 记录log
        self.history["loss/train"].append(loss)
        self.history["acc/train"].append(acc)
        self.history["f1/train"].append(f1)

    def validation_step(self, batch, idx):
        x, y = batch
        _y = self(x)
        val_loss = self.val_criterion(_y, y)
        self.val_metric.update(_y, y)
        return val_loss

    def validation_epoch_end(self, outs):
        val_loss = sum(outs).item() / len(outs)
        val_acc, val_f1 = self.val_metric.compute()

        self.history["loss/test"].append(val_loss)
        self.history["acc/test"].append(val_acc)
        self.history["f1/test"].append(val_f1)

    def configure_optimizers(self):
        # 设置优化器
        return Adam(self.parameters())

model = CustomModel()


# In[ ]:




