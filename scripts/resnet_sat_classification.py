#!/usr/bin/env python
# coding: utf-8

# ## reference
# https://colab.research.google.com/drive/1-MAcNk-feD-k_oq3U9wxxI98gh2V5P9B?authuser=1#scrollTo=vaRmKAViVQfQ
# https://github.com/Stevellen/ResNet-Lightning/blob/master/resnet_classifier.py

# get_ipython().system('kaggle datasets download -d mahmoudreda55/satellite-image-classification')
# get_ipython().system('unzip satellite-image-classification.zip')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.optim import SGD, Adam
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
root_dir = 'data'
valid_split = 0.2
# In[6]:


# Here we define a new class to turn the ResNet model that we want to use as a feature extractor
# into a pytorch-lightning module so that we can take advantage of lightning's Trainer object.
# We aim to make it a little more general by allowing users to define the number of prediction classes.
class ResNetClassifier(pl.LightningModule):
    def __init__(self, num_classes, resnet_version,
                train_path, vld_path, test_path=None, 
                optimizer='adam', lr=1e-3, batch_size=16,
                transfer=True, tune_fc_only=True):
        super().__init__()

        self.__dict__.update(locals())
        resnets = {
            18: models.resnet18, 34: models.resnet34,
            50: models.resnet50, 101: models.resnet101,
            152: models.resnet152
        }
        optimizers = {'adam': Adam, 'sgd': SGD}
        self.optimizer = optimizers[optimizer]
        #instantiate loss criterion
        self.criterion = nn.BCEWithLogitsLoss() if num_classes == 2 else nn.CrossEntropyLoss()
        # Using a pretrained ResNet backbone
        self.resnet_model = resnets[resnet_version](pretrained=transfer)
        # Replace old FC layer with Identity so we can train our own
        linear_size = list(self.resnet_model.children())[-1].in_features
        # replace final layer for fine tuning
        self.resnet_model.fc = nn.Linear(linear_size, num_classes)

        if tune_fc_only: # option to only tune the fully-connected layers
            for child in list(self.resnet_model.children())[:-1]:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, X):
        return self.resnet_model(X)

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)
    
    def train_dataloader(self):
        # values here are specific to pneumonia dataset and should be changed for custom data
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            transforms.RandomRotation(degrees=(30, 70)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


        img_train = ImageFolder(root_dir, transform=transform)

        dataset = ImageFolder(root_dir, transform=train_transform)
        dataset_test = ImageFolder(root_dir, transform=valid_transform)
        # print(f"Classes: {dataset.classes}")
        dataset_size = len(dataset)
        # print(f"Total number of images: {dataset_size}")
        valid_size = int(valid_split*dataset_size)
        # training and validation sets
        indices = torch.randperm(len(dataset)).tolist()
        dataset_train = Subset(dataset, indices[:-valid_size])
        return DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=4)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        if self.num_classes == 2:
            y = F.one_hot(y, num_classes=2).float()
        
        loss = self.criterion(preds, y)
        acc = (torch.argmax(y,1) == torch.argmax(preds,1))                 .type(torch.FloatTensor).mean()
        # perform logging
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def val_dataloader(self):
        # values here are specific to pneumonia dataset and should be changed for custom data
        transform = transforms.Compose([
                transforms.Resize((500,500)),
                transforms.ToTensor(),
                transforms.Normalize((0.48232,), (0.23051,))
        ])
        
        img_val = ImageFolder(self.vld_path, transform=transform)
        
        return DataLoader(img_val, batch_size=1, shuffle=False)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        if self.num_classes == 2:
            y = F.one_hot(y, num_classes=2).float()
        
        loss = self.criterion(preds, y)
        acc = (torch.argmax(y,1) == torch.argmax(preds,1))                 .type(torch.FloatTensor).mean()
        # perform logging
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, logger=True)


    def test_dataloader(self):
        # values here are specific to pneumonia dataset and should be changed for custom data
        transform = transforms.Compose([
                transforms.Resize((500,500)),
                transforms.ToTensor(),
                transforms.Normalize((0.48232,), (0.23051,))
        ])
        
        img_test = ImageFolder(self.test_path, transform=transform)
        
        return DataLoader(img_test, batch_size=1, shuffle=False)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        if self.num_classes == 2:
            y = F.one_hot(y, num_classes=2).float()
        
        loss = self.criterion(preds, y)
        acc = (torch.argmax(y,1) == torch.argmax(preds,1))                 .type(torch.FloatTensor).mean()
        # perform logging
        self.log("test_loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("test_acc", acc, on_step=True, prog_bar=True, logger=True)


# 

# In[ ]:


model = ResNetClassifier(num_classes = 4, resnet_version = 34,
                        train_path = args.train_set, vld_path = args.vld_set, test_path = args.test_set,
                        transfer = args.transfer, tune_fc_only = args.tune_fc_only)
# Instantiate lightning trainer and train model
trainer_args = {'gpus': args.gpus, 'max_epochs': args.num_epochs}
trainer = pl.Trainer(**trainer_args)
trainer.fit(model)
# Save trained model
save_path = (args.save_path if args.save_path is not None else '/') + 'trained_model.ckpt'
trainer.save_checkpoint(save_path)

