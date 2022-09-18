#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/suredream/PyTorch-Lightning-Ace/blob/main/pll-transfer-learning.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Transfer Learning from Supervised and Self-Supervised Pretraining using PyTorch Lightning
# 
# - toc: true
# - category: blog
# - tags: pytorch-lightning transfer-learning supervised-learning self-supervised-learning

# Credit to original author William Falcon, and also to Alfredo Canziani for posting the video presentation: [_Supervised and self-supervised transfer learning (with PyTorch Lightning)_](https://www.youtube.com/watch?v=nCq_vy9qE-k)
# 
# In the video presentation, they compare transfer learning from pretrained:
# * supervised
# * self-supervised
# 
# However, I would like to point out that the comparison is not entirely fair for the case of supervised pretraining. The reason is that they do not replace the last fully-connected layer of the supervised pretrained backbone model with the new finetuning layer. Instead, they stack the new finetuning layer on top of the pretrained model (including its last fully connected layer).
# 
# This is a clear disadvantage for the supervised pretrained model because:
# * all its expressive power is contained in the output of the penultimate layer
# * and it was already used by the last fully-connected layer to predict 1,000 classes
# 
# When stacking the finetuning layer on top of it, this has to perform the 10-class classification using the output of the 1,000-class classfication layer.
# 
# On the contrary, if we replace the backbone last fully connected layer with the new finetuning layer, it will be able to perform the 10-class classification using all the expressive power of the features coming from the output of the penultimate layer.
# 
# In this notebook I show that if we replace the last fully connected layer with the new finetuning layer, both supervised and self-supervised approaches give comparable results.

# In[3]:


# %%capture
get_ipython().system('pip install pytorch-lightning')
get_ipython().system('pip install pytorch-lightning-bolts')


# In[4]:


import pytorch_lightning as pl
import pl_bolts

print(f"pl version: {pl.__version__}")
print(f"pl_bolts version: {pl_bolts.__version__}")


# In[5]:


import torch
from torchvision import models

resnet50 = models.resnet50(pretrained=True)


# In[6]:


from torchvision.datasets import CIFAR10
from torchvision import transforms

normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                 std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

cf10_transforms = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

cifar_10 = CIFAR10('.', train=True, download=True, transform=cf10_transforms)


# In[7]:


from matplotlib import pyplot as plt

image, label = next(iter(cifar_10))
print(f"LABEL: {label}")
plt_img = image.numpy().transpose(1, 2, 0)
plt.imshow(plt_img);


# In[8]:


from torch.utils.data import DataLoader

train_loader = DataLoader(cifar_10, batch_size=32, shuffle=True)


# In[9]:


for batch in train_loader:
    x, y = batch
    print(x.shape, y.shape)
    break


# In[10]:


import torch
from torchvision import models

resnet50 = models.resnet50(pretrained=True)

for param in resnet50.parameters():
    param.requires_grad = False

num_classes = 10
resnet50.fc = torch.nn.Linear(resnet50.fc.in_features, num_classes)

# Use afterwards in optimizer: resnet50.fc.parameters()


# In[11]:


x, y = next(iter(train_loader))

preds = resnet50(x)
preds[:5]


# In[12]:


from torch.nn.functional import softmax

preds = softmax(preds, dim=-1)
preds[:5]


# In[13]:


pred_labels = torch.argmax(preds, dim=-1)
pred_labels[:5]


# In[14]:


y[:5]


# In[15]:


#  Bolts: Data Module: 3 data loaders

from pl_bolts.datamodules import CIFAR10DataModule

dm = CIFAR10DataModule('.')


# ## Supervised Pretraining

# ### Fitting only the new finetuning layer

# In[16]:


# # PyTorch

# from torch.nn.functional import cross_entropy
# from torch.optim import Adam

# optimizer = Adam(resnet50.fc.parameters(), lr=1e-3)

# epochs = 10
# for epoch in range(epochs):
#     for batch in dm.train_dataloader():
#         x, y = batch

#         # features = backbone(x)
#         # # disable gradients to backbone if all parameters used by the optimizer
#         # features = features.detach()

#         # # tell PyTorch not to track the computational graph: much faster, less memory used: not backpropagated
#         # with torch.no_grad():
#         #     features = backbone(x)

#         # preds = finetune_layer(features)

#         preds = resnet50(x)

#         loss = cross_entropy(preds, y)

#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()


# In[17]:


# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy

from torch.nn.functional import cross_entropy
from torch.optim import Adam

class ImageClassifier(pl.LightningModule):
    def __init__(self, num_classes=10, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        # self.num_classes = num_classes
        # self.lr = lr

        self.model = models.resnet50(pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)

    def training_step(self, batch, batch_idx):
        # return the loss given a batch: this has a computational graph attached to it: optimization
        x, y = batch
        preds = self.model(x)
        loss = cross_entropy(preds, y)
        self.log('train_loss', loss)  # lightning detaches your loss graph and uses its value
        self.log('train_acc', accuracy(preds, y))
        return loss

    def configure_optimizers(self):
        # return optimizer
        optimizer = Adam(self.model.fc.parameters(), lr=self.hparams.lr)
        return optimizer


# In[ ]:


classifier = ImageClassifier()

trainer = pl.Trainer(progress_bar_refresh_rate=20, gpus=1, max_epochs=2)  # for Colab: set refresh rate to 20 instead of 10 to avoid freezing
trainer.fit(classifier, dm)  # train_loader


# In[ ]:


# Start tensorboard
get_ipython().run_line_magic('reload_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', '--logdir lightning_logs/')


# ![](img/20201117-1-train_acc.svg "Train Accuracy")
# 
# ![](img/20201117-1-train_loss.svg "Train Loss")

# ### Fitting all the model after 10 epochs

# In[18]:


# PyTorch Lightning
import pytorch_lightning as pl
# from pytorch_lightning.metrics.functional import accuracy
import torchmetrics

from torch.nn.functional import cross_entropy
from torch.optim import Adam

class ImageClassifier(pl.LightningModule):
    def __init__(self, num_classes=10, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        # self.num_classes = num_classes
        # self.lr = lr

        self.model = models.resnet50(pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)

    def training_step(self, batch, batch_idx):
        # return the loss given a batch: this has a computational graph attached to it: optimization
        x, y = batch
        if self.trainer.current_epoch == 10:
            for param in self.model.parameters():
                param.requires_grad = True
        preds = self.model(x)
        loss = cross_entropy(preds, y)
        self.log('train_loss', loss)  # lightning detaches your loss graph and uses its value
        self.log('train_acc', torchmetrics.functional.accuracy(preds, y))
        return loss

    def configure_optimizers(self):
        # return optimizer
        optimizer = Adam(self.model.parameters(), lr=self.hparams.lr)  # self.model.fc.parameters()
        return optimizer


# In[19]:


classifier = ImageClassifier()

trainer = pl.Trainer(progress_bar_refresh_rate=5, gpus=1, limit_train_batches=20, max_epochs=20)
trainer.fit(classifier, dm)  # train_loader


# In[20]:


# Start tensorboard
get_ipython().run_line_magic('reload_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', '--logdir lightning_logs/')


# ![](img/20201117-2-train_acc.svg "Train Accuracy")
# 
# ![](img/20201117-2-train_loss.svg "Train Loss")

# ## Self-Supervised Pretraining
# 
# https://pytorch-lightning-bolts.readthedocs.io/en/latest/self_supervised_models.html#swav

# ### Fitting all the model after 10 epochs

# In[22]:


# PyTorch Lightning
import pytorch_lightning as pl
# from pytorch_lightning.metrics.functional import accuracy
import torchmetrics
from torch.nn.functional import cross_entropy
from torch.optim import Adam

from pl_bolts.models.self_supervised import SwAV
weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/swav/swav_imagenet/swav_imagenet.pth.tar'
swav = SwAV.load_from_checkpoint(weight_path, strict=True)

# from pl_bolts.models.self_supervised import SimCLR
# weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/simclr-cifar10-v1-exp12_87_52/epoch%3D960.ckpt'
# simclr = SimCLR.load_from_checkpoint(weight_path, strict=False)


class ImageClassifier(pl.LightningModule):
    def __init__(self, num_classes=10, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        # self.num_classes = num_classes
        # self.lr = lr

        # self.model = models.resnet50(pretrained=True)
        self.backbone = swav.model
        # self.backbone = simclr

        for param in self.backbone.parameters():
            param.requires_grad = False

        # self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
        self.finetune_layer = torch.nn.Linear(3000, num_classes)

    def training_step(self, batch, batch_idx):
        # return the loss given a batch: this has a computational graph attached to it: optimization
        x, y = batch
        if self.trainer.current_epoch == 10:
            for param in self.backbone.parameters():
                param.requires_grad = True
        (features1, features2) = self.backbone(x)
        features = features2
        # features = self.backbone(x)
        preds = self.finetune_layer(features)
        loss = cross_entropy(preds, y)
        self.log('train_loss', loss)  # lightning detaches your loss graph and uses its value
        self.log('train_acc', torchmetrics.functional.accuracy(preds, y))
        return loss

    def configure_optimizers(self):
        # return optimizer
        optimizer = Adam(self.parameters(), lr=self.hparams.lr)  # self.model.fc.parameters()
        return optimizer


# In[23]:


classifier = ImageClassifier()

trainer = pl.Trainer(progress_bar_refresh_rate=5, gpus=1, limit_train_batches=20, max_epochs=20)
trainer.fit(classifier, dm)  # train_loader


# In[24]:


# Start tensorboard
get_ipython().run_line_magic('reload_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', '--logdir lightning_logs/')


# ![](img/20201117-3-train_acc.svg "Train Accuracy")
# 
# ![](img/20201117-3-train_loss.svg "Train Loss")
