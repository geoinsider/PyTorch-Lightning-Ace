#!/usr/bin/env python
# coding: utf-8
import pytorch_lightning as pl
import os
import numpy as np 
import random
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from argparse import Namespace


# In[ ]:


"""
from google.colab import drive
drive.mount('./content/drive')

import os
os.chdir("/content/drive/My Drive/")
"""


# In[ ]:


from pytorch_lightning import seed_everything

# Set seed
seed = 42
seed_everything(seed)


# In[ ]:


class CoolSystem(pl.LightningModule):

    def __init__(self, hparams):
        super(CoolSystem, self).__init__()

        self.params = hparams
        
        self.data_dir = self.params.data_dir
        self.num_classes = self.params.num_classes 

        ########## define the model ########## 
        arch = torchvision.models.resnet18(pretrained=True)
        num_ftrs = arch.fc.in_features

        modules = list(arch.children())[:-1] # ResNet18 has 10 children
        self.backbone = torch.nn.Sequential(*modules) # [bs, 512, 1, 1]
        self.final = torch.nn.Sequential(
               torch.nn.Linear(num_ftrs, 128),
               torch.nn.ReLU(inplace=True),
               torch.nn.Linear(128, self.num_classes),
               torch.nn.Softmax(dim=1))

    def forward(self, x):
        x = self.backbone(x)
        x = x.reshape(x.size(0), -1)
        x = self.final(x)
        
        return x
    
    def configure_optimizers(self):
        # REQUIRED
        optimizer = torch.optim.SGD([
                {'params': self.backbone.parameters()},
                {'params': self.final.parameters(), 'lr': 1e-2}
            ], lr=1e-3, momentum=0.9)

        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
 
        return [optimizer], [exp_lr_scheduler]

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x)
        
        loss = F.cross_entropy(y_hat, y)
        
        _, preds = torch.max(y_hat, dim=1)
        acc = torch.sum(preds == y.data) / (y.shape[0] * 1.0)

        self.log('train_loss', loss)
        self.log('train_acc', acc)

        return {'loss': loss, 'train_acc': acc}
    

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        _, preds = torch.max(y_hat, 1)
        acc = torch.sum(preds == y.data) / (y.shape[0] * 1.0)

        self.log('val_loss', loss)
        self.log('val_acc', acc)

        return {'val_loss': loss, 'val_acc': acc}


    def test_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        _, preds = torch.max(y_hat, 1)
        acc = torch.sum(preds == y.data) / (y.shape[0] * 1.0)

        return {'test_loss': loss, 'test_acc': acc}


    def train_dataloader(self):
        # REQUIRED

        transform = transforms.Compose([
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])

        train_set = torchvision.datasets.ImageFolder(os.path.join(self.data_dir, 'train'), transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)

        return train_loader
    
    def val_dataloader(self):
      transform = transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])
                              
      val_set = torchvision.datasets.ImageFolder(os.path.join(self.data_dir, 'val'), transform)
      val_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=True, num_workers=4)

      return val_loader

    def test_dataloader(self):
      transform = transforms.Compose([
                              transforms.Resize(256),
                              transforms.CenterCrop(224),
                              transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                              ])
                            
      val_set = torchvision.datasets.ImageFolder(os.path.join(self.data_dir, 'val'), transform)
      val_loader = torch.utils.data.DataLoader(val_set, batch_size=8, shuffle=True, num_workers=4)

      return val_loader


# In[ ]:


def main(hparams):
  model = CoolSystem(hparams)


  trainer = pl.Trainer(
      max_epochs=hparams.epochs,
      gpus=1,
      accelerator='dp'
  )  

  trainer.fit(model)


# In[ ]:


args = {
    'num_classes': 2,
    'epochs': 5,
    'data_dir': "/content/hymenoptera_data",
}

hyperparams = Namespace(**args)

main(hyperparams)


# In[ ]:


# resume training

RESUME = True

if RESUME:
    resume_checkpoint_dir = './lightning_logs/version_0/checkpoints/'
    checkpoint_path = os.listdir(resume_checkpoint_dir)[0]
    resume_checkpoint_path = resume_checkpoint_dir + checkpoint_path


    args = {
    'num_classes': 2,
    'data_dir': "/content/hymenoptera_data"}

    hparams = Namespace(**args)

    model = CoolSystem(hparams)

    
    trainer = pl.Trainer(gpus=1, 
                max_epochs=10,             
                accelerator='dp',
                resume_from_checkpoint = resume_checkpoint_path)

    trainer.fit(model)


# In[ ]:


# tensorboard

# get_ipython().run_line_magic('load_ext', 'tensorboard')
# get_ipython().run_line_magic('tensorboard', '--logdir=./lightning_logs')


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

# functions to show an image
def imshow(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.show()

classes = ['ants', 'bees']

checkpoint_dir = 'lightning_logs/version_1/checkpoints/'
checkpoint_path = checkpoint_dir + os.listdir(checkpoint_dir)[0]

checkpoint = torch.load(checkpoint_path)
model_infer = CoolSystem(hparams)
model_infer.load_state_dict(checkpoint['state_dict'])

try_dataloader = model_infer.test_dataloader()

inputs, labels = next(iter(try_dataloader))

# print images and ground truth
imshow(torchvision.utils.make_grid(inputs))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(8)))

# inference
outputs = model_infer(inputs)

_, preds = torch.max(outputs, dim=1)
# print (preds)
print (torch.sum(preds == labels.data) / (labels.shape[0] * 1.0))

print('Predicted: ', ' '.join('%5s' % classes[preds[j]] for j in range(8)))

