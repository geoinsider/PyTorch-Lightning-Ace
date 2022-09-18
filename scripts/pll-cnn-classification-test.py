#!/usr/bin/env python
# coding: utf-8

import os
import shutil
import opendatasets as od
import pandas as pd
import numpy as np
from PIL import Image

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torchvision.utils import make_grid
from torchmetrics.functional import accuracy
# from image_classifier import CNNImageClassifier

import pytorch_lightning as pl


# In[6]:


print("pandas version:",pd.__version__)
print("numpy version:",np.__version__)
#print("seaborn version:",sns.__version__)
print("torch version:",torch.__version__)
print("pytorch ligthening version:",pl.__version__)

df_labels = pd.read_csv('histopathologic-cancer-detection/train_labels.csv')
df_labels.head()


# In[16]:


df_labels['label'].value_counts()


# There are 130908 normal cases (0) and and 89117 abnormal (cancerous) cases (1) 
# 
# *   List item
# *   List item
# 
# which is not highly unbalanced. 

# In[17]:


print('No. of images in training dataset: ', len(os.listdir("histopathologic-cancer-detection/train")))
print('No. of images in testing dataset: ', len(os.listdir("histopathologic-cancer-detection/test")))


# This is a huge dataset which requires a lot of compute time and resources so for the purpose of learning our first basic image classification model, we will downsample it to 5000 images and then split it into training and testing dataset.

# In[18]:


# Setting seed to make the results replicable
np.random.seed(0)
train_imgs_orig = os.listdir("histopathologic-cancer-detection/train")
selected_image_list = []
for img in np.random.choice(train_imgs_orig, 10000):
  selected_image_list.append(img)
len(selected_image_list)


# In[19]:


selected_image_list[0]


# In[20]:


fig = plt.figure(figsize=(25, 6))
for idx, img in enumerate(np.random.choice(selected_image_list, 20)):
    ax = fig.add_subplot(2, 20//2, idx+1, xticks=[], yticks=[])
    im = Image.open("histopathologic-cancer-detection/train/" + img)
    plt.imshow(im)
    lab = df_labels.loc[df_labels['id'] == img.split('.')[0], 'label'].values[0]
    ax.set_title(f'Label: {lab}')


# In[21]:


np.random.seed(0)
np.random.shuffle(selected_image_list)
train_idx = selected_image_list[:8000]
test_idx = selected_image_list[8000:]
print("Number of images in the downsampled training dataset: ", len(train_idx))
print("Number of images in the downsampled testing dataset: ", len(test_idx))


# ### Processing the dataset
# The following information has been provided on the Kaggle and the Github where the dataset is hosted - "A positive label indicates that the center 32x32px region of a patch contains at least one pixel of tumor tissue. Tumor tissue in the outer region of the patch does not influence the label. This outer region is provided to enable the design of fully-convolutional models that do not use any zero-padding, to ensure consistent behavior when applied to a whole-slide image."

# In[22]:


# from google.colab import drive
# drive.mount('/content/gdrive')


# In[23]:


#os.mkdir('/content/histopathologic-cancer-detection/train_dataset/')


# In[24]:


os.mkdir('/content/histopathologic-cancer-detection/test_dataset/')
for fname in test_idx:
  src = os.path.join('histopathologic-cancer-detection/train', fname)
  dst = os.path.join('/content/histopathologic-cancer-detection/test_dataset/', fname)
  shutil.copyfile(src, dst)
print('No. of images in downsampled testing dataset: ', len(os.listdir("/content/histopathologic-cancer-detection/test_dataset/")))


# In[25]:


os.mkdir('/content/histopathologic-cancer-detection/train_dataset/')
for fname in train_idx:
  src = os.path.join('histopathologic-cancer-detection/train', fname)
  dst = os.path.join('/content/histopathologic-cancer-detection/train_dataset/', fname)
  shutil.copyfile(src, dst)
print('No. of images in downsampled training dataset: ', len(os.listdir("/content/histopathologic-cancer-detection/train_dataset/")))


# In[26]:


data_T_train = T.Compose([
    T.CenterCrop(32),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.ToTensor(),
    ])
data_T_test = T.Compose([
    T.CenterCrop(32),
    T.ToTensor(),
    ])


# In[27]:


# Extracting the labels for the images that were selected in the downsampled data
selected_image_labels = pd.DataFrame()
id_list = []
label_list = []

for img in selected_image_list:
  label_tuple = df_labels.loc[df_labels['id'] == img.split('.')[0]]
  id_list.append(label_tuple['id'].values[0])
  label_list.append(label_tuple['label'].values[0])


# In[28]:


selected_image_labels['id'] = id_list
selected_image_labels['label'] = label_list
selected_image_labels.head()


# In[29]:


# dictionary with labels and ids of train data
img_label_dict = {k:v for k, v in zip(selected_image_labels.id, selected_image_labels.label)}


# Pytorch lightning expects data to be in folders with the classes. We cannot use the DataLoader module directly when all train images are in one folder without subfolders. So, we will write our custom function to carry out the loading. 
# 

# In[30]:


class LoadCancerDataset(Dataset):
    def __init__(self, data_folder, 
                 transform = T.Compose([T.CenterCrop(32),T.ToTensor()]), dict_labels={}):
        self.data_folder = data_folder
        self.list_image_files = [s for s in os.listdir(data_folder)]
        self.transform = transform
        self.dict_labels = dict_labels
        self.labels = [dict_labels[i.split('.')[0]] for i in self.list_image_files]

    def __len__(self):
        return len(self.list_image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_folder, self.list_image_files[idx])
        image = Image.open(img_name)
        image = self.transform(image)
        img_name_short = self.list_image_files[idx].split('.')[0]

        label = self.dict_labels[img_name_short]
        return image, label


# In[31]:
# Load train data 
train_set = LoadCancerDataset(data_folder='/content/histopathologic-cancer-detection/train_dataset/', 
                        # datatype='train', 
                        transform=data_T_train, dict_labels=img_label_dict)

# In[32]:


test_set = LoadCancerDataset(data_folder='/content/histopathologic-cancer-detection/test_dataset/', 
                         transform=data_T_test, dict_labels=img_label_dict)


# In[33]:


batch_size = 256

train_dataloader = DataLoader(train_set, batch_size, num_workers=2, pin_memory=True, shuffle=True)
test_dataloader = DataLoader(test_set, batch_size, num_workers=2, pin_memory=True)


# In[34]:


class CNNImageClassifier(pl.LightningModule):

    def __init__(self, learning_rate = 0.001):
        super().__init__()

        self.learning_rate = learning_rate

        self.conv_layer1 = nn.Conv2d(in_channels=3,out_channels=3,kernel_size=3,stride=1,padding=1)
        self.relu1=nn.ReLU()
        self.pool=nn.MaxPool2d(kernel_size=2)
        self.conv_layer2 = nn.Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=1)
        self.relu2=nn.ReLU()
        self.fully_connected_1 =nn.Linear(in_features=16 * 16 * 6,out_features=1000)
        self.fully_connected_2 =nn.Linear(in_features=1000,out_features=500)
        self.fully_connected_3 =nn.Linear(in_features=500,out_features=250)
        self.fully_connected_4 =nn.Linear(in_features=250,out_features=120)
        self.fully_connected_5 =nn.Linear(in_features=120,out_features=60)
        self.fully_connected_6 =nn.Linear(in_features=60,out_features=2)
        self.loss = nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def forward(self, input):
        output=self.conv_layer1(input)
        output=self.relu1(output)
        output=self.pool(output)
        output=self.conv_layer2(output)
        output=self.relu2(output)
        output=output.view(-1, 6*16*16)
        output = self.fully_connected_1(output)
        output = self.fully_connected_2(output)
        output = self.fully_connected_3(output)
        output = self.fully_connected_4(output)
        output = self.fully_connected_5(output)
        output = self.fully_connected_6(output)
        return output

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs) 
        train_accuracy = accuracy(outputs, targets)
        loss = self.loss(outputs, targets)
        self.log('train_accuracy', train_accuracy, prog_bar=True)
        self.log('train_loss', loss)
        return {"loss":loss, "train_accuracy":train_accuracy}

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs)
        test_accuracy = accuracy(outputs, targets)
        loss = self.loss(outputs, targets)
        self.log('test_accuracy', test_accuracy)
        return {"test_loss":loss, "test_accuracy":test_accuracy}

    def configure_optimizers(self):
        params = self.parameters()
        optimizer = optim.Adam(params=params, lr = self.learning_rate)
        return optimizer

    # Calculate accuracy for each batch at a time
    def binary_accuracy(self, outputs, targets):
        _, outputs = torch.max(outputs,1)
        correct_results_sum = (outputs == targets).sum().float()
        acc = correct_results_sum/targets.shape[0]
        return acc

    def predict_step(self, batch, batch_idx ):
        return self(batch)


# In[35]:


model = CNNImageClassifier()

trainer = pl.Trainer(fast_dev_run=True, gpus=1)
trainer.fit(model, train_dataloaders=train_dataloader)


# In[36]:


ckpt_dir = "/content/gdrive/MyDrive/Colab Notebooks/cnn"
ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_epochs=25)

model = CNNImageClassifier()
trainer = pl.Trainer(
    default_root_dir=ckpt_dir,
                     gpus=-1,
                    #  progress_bar_refresh_rate=30,
                        callbacks=[ckpt_callback],
                        log_every_n_steps=25,
                        max_epochs=500)
trainer.fit(model, train_dataloaders=train_dataloader)


# In[37]:


trainer.test(dataloaders=test_dataloader)


# In[38]:


model.eval()
preds = []
for batch_i, (data, target) in enumerate(test_dataloader):
    data, target = data.cuda(), target.cuda()
    output = model.cuda()(data)

    pr = output[:,1].detach().cpu().numpy()
    for i in pr:
        preds.append(i)


# In[39]:


test_preds = pd.DataFrame({'imgs': test_set.list_image_files, 'labels':test_set.labels,  'preds': preds})


# In[40]:


test_preds['imgs'] = test_preds['imgs'].apply(lambda x: x.split('.')[0])


# In[41]:


test_preds.head()


# In[42]:


test_preds['predictions'] = 1
test_preds.loc[test_preds['preds'] < 0, 'predictions'] = 0
test_preds.shape


# In[43]:


test_preds.head()


# In[44]:


len(np.where(test_preds['labels'] == test_preds['predictions'])[0])/test_preds.shape[0]


# In[44]:




