#!/usr/bin/env python
# coding: utf-8

# In[61]:


# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, Sigmoid, functional as F
from torch.optim import Adam, SGD, RMSprop
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets


# In[87]:


import os, shutil, pickle, time, itertools
import pandas as pd
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef
from collections import OrderedDict
from pathlib import Path


# In[91]:


data_dir = "input_data_repeat_classification/"


# In[89]:


numImages = sum([len(files) for r, d, files in os.walk(data_dir)])


# In[80]:


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


# In[90]:


BATCH_SIZE = 1
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

dataset = ImageFolderWithPaths(data_dir,
                         transforms.Compose([
                             transforms.Resize((256,256)),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             normalize,
                         ]))

data_loader = Data.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=2)


# In[82]:


class Net(Module):   
    def __init__(self):
        super(Net, self).__init__()
    
        # input 256 x 256
        self.layer1 = Sequential(
            # Defining a 2D convolution layer
            Conv2d(3, 32, kernel_size=3, stride=1, padding=1), # 256 x 256
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2)
        )
        
        # out size : 128 x 128

        self.layer2 = Sequential(
            # Defining a 2D convolution layer
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2) 
        )
        
        # out size : 64 x 64
        
        self.layer3 = Sequential(
            # Defining a 2D convolution layer
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2) 
        )
        
        #out size : 32 x 32
        
        self.drop_out = Dropout(0.2)
        
        self.fc1 = Linear(32*32*32 , 256)
        
        self.fc2 = Linear(256, 128)
        
        self.fc3 = Linear(128, 3)
        
        self.sigmoid = Sigmoid()


    # Defining the forward pass    
    def forward(self, x):
        x = self.layer1(x)
        x = F.dropout(x, p=0.2)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        #x = F.dropout(x, p=0.2)
        x = self.fc1(x)
        x = F.dropout(x, p=0.2)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        #x = self.sigmoid(x)
        x = F.softmax(x, dim=1)

        return x


# In[83]:


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy_list(y_pred, y_actual, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = y_actual.size(0)

    _, pred = y_pred.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(y_actual.view(1, -1).expand_as(pred))
#     print ("correct : ", correct)
#     print ("y_actual : ", y_actual)
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        #print ("correct_k : ", correct_k)
        #print (correct_k.mul_(100.0 / batch_size))
        res.append(correct_k.mul_(100.0 / batch_size))

    return [correct, res]

def accuracy(y_pred, y_actual, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = y_actual.size(0)

    _, pred = y_pred.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(y_actual.view(1, -1).expand_as(pred))
#     print ("correct : ", correct)
#     print ("y_actual : ", y_actual)
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        #print ("correct_k : ", correct_k)
        #print (correct_k.mul_(100.0 / batch_size))
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


# In[84]:


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model = Net()
learning_rate=0.0001

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.DataParallel(model, device_ids=[x for x in range(torch.cuda.device_count())])

model.to(device)

model_path = "saved_models/repeat_type.pth.tar"
load_checkpoint = True
if os.path.isfile(model_path) and load_checkpoint: 
    print("=> loading checkpoint '{}'".format(model_path))
    checkpoint = torch.load(model_path, map_location=device)
    best_acc1 = checkpoint['best_acc1']
    if device.type == 'cpu':
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[7:] # remove module
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
         model.load_state_dict(checkpoint['state_dict'])
    print ("=> Model loaded from epoch :", checkpoint['epoch'], "with accuracy :", best_acc1)
else:
    print("=> no model found at '{}'".format(model_path))


# In[85]:


test_pred = torch.tensor([], dtype=torch.long, device="cpu")
test_metadata = []
output_with_label = []

for idx, data in enumerate(data_loader):
    inputs, labels, metadata = data
    inputs, labels = Variable(inputs), Variable(labels)     
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs)
    test_metadata += metadata
    test_pred = torch.cat((test_pred, outputs.argmax(dim=1).cpu()), 0)
    
for i in range(len(test_metadata)):
    output_with_label.append([str(test_metadata[i].split("/")[-1]), test_pred[i].item()])


# In[86]:


class_dic = {0:"Class III", 1:"Class IV", 2:"Non-repeat"}
for i in output_with_label:
    print (i[0], ": ", class_dic[i[1]])


# In[ ]:




