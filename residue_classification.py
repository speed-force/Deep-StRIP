#!/usr/bin/env python
# coding: utf-8

# # make python script for github from this code

# In[1]:


# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, Sigmoid, functional as F
from torch.optim import Adam, SGD, RMSprop
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets


# In[2]:


import os, shutil, pickle, json, time, itertools, copy
from math import sqrt
import pandas as pd
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef,r2_score
from torchsummary import summary
from collections import OrderedDict
import warnings


# In[3]:


data_path = "input_data_residue_classification/"


# In[4]:


class ImageFolderWithPaths(datasets.ImageFolder):
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


# In[5]:


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
no_workers = 2
BATCH_SIZE = 64

test_dataset = ImageFolderWithPaths(data_path,
                         transforms.Compose([
                             transforms.Resize((32,256)),
                             transforms.ToTensor(),
                             normalize,
                         ]))

test_loader = Data.DataLoader(
            test_dataset,
    batch_size=BATCH_SIZE,
    pin_memory=True,
    num_workers=no_workers)



# In[6]:


class Net(Module):   
    def __init__(self):
        super(Net, self).__init__()
    
        # input 256 x 256
        self.layer1 = Sequential(
            Conv2d(3, 32, kernel_size=3, stride=1, padding=1), # 256 x 256
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2)
        )
        
        # out size : 128 x 128
        self.layer2 = Sequential(
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2) 
        )
        
        # out size : 64 x 64
        self.layer3 = Sequential(
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2) 
        )
        
        #out size : 32 x 32
        
        self.drop_out = Dropout(0.2)
        
        self.fc1 = Linear(32*32*4 , 256)
        
        self.fc2 = Linear(256, 128)
        
        self.fc3 = Linear(128, 2)
        
        self.sigmoid = Sigmoid()


    # Defining the forward pass    
    def forward(self, x):
        x = self.layer1(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x


# In[7]:


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model = Net()
learning_rate=0.0001

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.DataParallel(model, device_ids=[x for x in range(torch.cuda.device_count())])

model.to(device)

model_path = "saved_models/repeat_region.pth.tar"
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
    print("=> no checkpoint found at '{}'".format(model_path))

model.eval()


# In[8]:


test_pred = torch.tensor([], dtype=torch.long, device="cpu")
test_true = torch.tensor([], dtype=torch.long, device="cpu")
test_metadata = []
output_with_label = []
for idx, data in enumerate(test_loader):
    inputs, labels, metadata = data
    inputs, labels = Variable(inputs), Variable(labels)     
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs)
    test_metadata += metadata
    test_pred = torch.cat((test_pred, outputs.argmax(dim=1).cpu()), 0)
    test_true = torch.cat((test_true, labels.cpu()), 0)
    outputs = outputs.to(device)

for i in range(len(test_metadata)):
    output_with_label.append([str(test_metadata[i].split("/")[-1]), test_pred[i].item(), test_true[i].item()])


# In[9]:


def get_pdb_label_dic(output_with_label):
    mapper = {0:"Non Repeating", 1:"Repeating"}
    pdb_label_dic = {}
    tot_acc = 0

    for i in output_with_label:
        pdb = i[0].split('_')[0]
        rep_no = int(i[0].split('_')[1].split('.')[0])
        if pdb not in pdb_label_dic.keys():
            pdb_label_dic[pdb] = []
        i.append(rep_no)
        pdb_label_dic[pdb].append(i)


    for key in pdb_label_dic.keys():
        val = pdb_label_dic[key]
        val.sort(key = lambda x: x[3])
        for i in val:
            i.pop()
        pdb_label_dic[key] = val


    for key in pdb_label_dic.keys():
        val = pdb_label_dic[key]
        numFalse = 0
        total_count = len(val)
        acc_class = {}
        tot_class = {}
        for i in val:
            if i[1] != i[2]:
                numFalse += 1

            if i[1] == i[2]:
                if i[2] not in acc_class:
                    acc_class[i[2]] = 1
                else:
                    acc_class[i[2]] += 1

            if i[2] not in tot_class:
                tot_class[i[2]] = 1
            else:
                tot_class[i[2]] += 1
    return pdb_label_dic

def res_window(window,rep_tolerance):
        l=len(window)
        rep=sum(window)
        if rep/l > rep_tolerance:
                return 1
        return 0

def windowing(y,window_len,rep_tolerance):
        l=len(y)
        new_y=[]
        for i in range(l):
            left = i - window_len/2
            rt = i + window_len/2
            while (left < 0):
                left += 1
                rt += 1
            while (rt > l-1):
                rt -= 1
                left -= 1
            new_y.append(res_window(y[int(left):int(rt)+1],rep_tolerance))
        return new_y
    
def ranges(p1):
    p = []
    for i in p1:
        try:
            p.append(int(i))
        except:
            p.append(int(i[:-1]))
    q = sorted(p)
    i = 0
    if len(q) > 0:
        for j in range(1,len(q)):
            if q[j] > 1+q[j-1]:
                yield (str(q[i])+"-"+str(q[j-1]))
                i = j
        yield (str(q[i])+"-"+str(q[-1]))


# In[10]:


def print_repeat_regions(pdb_label_dic):
    total_true = []
    total_pred = []
    cnt = 0
    json_dic = {}
    output_dic = {} # repeat regions for all proteins
    for i in pdb_label_dic:
        
        if len(i) > 5:
            continue

        cnt += 1
        residue_list = numResid_pdb[i.upper()] #str list
        residue_list_int = []
        
        smoothed_rep_list = [] # str list
        for repeat in pdb_label_dic[i]:
            rep_no = repeat[0].split(".")[0].split("_")[1]
            if repeat[3] == 1:
                smoothed_rep_list.append(rep_no)
                
        new_rep_list = []
        min_i = 99999
        max_i = -99999
        for j in smoothed_rep_list:
            try:
                i_no = int(j)
            except:
                i_no = int(j[:-1])
            
            if i_no < min_i:
                min_i = i_no
            if i_no > max_i:
                max_i = i_no
                
        for j in residue_list:
            try:
                i_no = int(j)
            except:
                i_no = int(j[:-1])
                
            if i_no >= min_i and i_no <= max_i:
                new_rep_list.append(i_no)              

        sm_rep_range = ranges(new_rep_list)
        #print ("PDB chain: ", i)
        #print ("Predicted Region : ", end="")
        #for s in sm_rep_range:
        #    print (s+";",end="")
        #print ()
        output_dic[i] = new_rep_list

    #print ("Total number of Protein chains : ", cnt)
    return output_dic


# In[11]:


pdb_label_dic = get_pdb_label_dic(output_with_label)
pdb_dic = copy.deepcopy(pdb_label_dic)
for i in pdb_label_dic:
    pred_val = []
    for residue in pdb_label_dic[i]:
        pred_val.append(residue[1])
    smoothed_val = windowing(pred_val, 24, 0.7)
    for idx, repeat in enumerate(pdb_label_dic[i]):
        repeat.append(smoothed_val[idx])


# In[12]:


with open("util_files/residue_data.json", "r") as o:
    numResid_pdb = json.load(o)


# In[13]:


rep_regions_dic = print_repeat_regions(pdb_label_dic)


# ### Copy number classification 

# In[16]:


predres_folder = 'predicted_residues/'
if not os.path.isdir(predres_folder):
    os.mkdir(predres_folder)
    
for pdb in rep_regions_dic:
    pred_pdb_path = os.path.join(predres_folder, pdb)
    if not os.path.isdir(pred_pdb_path):
        os.mkdir(pred_pdb_path)
    if not len(os.listdir(pred_pdb_path)):
        pdb_path = os.path.join(data_path, pdb)
        if os.path.isdir(pdb_path):
            for residue in os.listdir(pdb_path):
                residue_path = os.path.join(pdb_path, residue)
                residue_no = int(residue.split('_')[1].split('.')[0])
                if residue_no in rep_regions_dic[pdb]:
                    os.system('cp '+residue_path+' '+pred_pdb_path)
                    
# Check only pred_res pdbs are present (otherwise delete)
for pdb_folder in os.listdir(predres_folder):
    if pdb_folder not in rep_regions_dic:
        os.system('rm -rf '+os.path.join(predres_folder, pdb_folder))


# In[18]:


data_path_cn = predres_folder
numImages = sum([len(files) for r, d, files in os.walk(data_path_cn)])


# In[19]:


BATCH_SIZE = 64
no_workers = 2
dataset_cn = ImageFolderWithPaths(data_path_cn,
                         transforms.Compose([
                             transforms.Resize((32,256)),
                             transforms.ToTensor(),
                             normalize]))

test_loader_cn = Data.DataLoader(
            dataset_cn,
            batch_size=BATCH_SIZE,
            pin_memory=True,
            num_workers=no_workers)
len(test_loader_cn)


# In[20]:


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model = Net()
learning_rate=0.0001

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.DataParallel(model, device_ids=[x for x in range(torch.cuda.device_count())])

model.to(device)

model_path = "saved_models/copy_number.pth.tar"
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
    print("=> no checkpoint found at '{}'".format(model_path))
    
model.eval()


# In[21]:


test_pred = torch.tensor([], dtype=torch.long, device="cpu")
test_metadata = []
output_with_label_rb = []
cnt = 1
for idx, data in enumerate(test_loader_cn):
#     print (cnt)
#     cnt += 1
    inputs, labels, metadata = data
    inputs, labels = Variable(inputs), Variable(labels)     
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs)
    test_metadata += metadata
    test_pred = torch.cat((test_pred, outputs.argmax(dim=1).cpu()), 0)

for i in range(len(test_metadata)):
    output_with_label_rb.append([str(test_metadata[i].split("/")[-1]), test_pred[i].item()])


# In[22]:


def get_pdb_label_dic_cn(output_with_label_rb):
    pdb_label_dic = {}

    for i in output_with_label_rb:
        pdb = i[0].split('_')[0].upper()
        rep_no = int(i[0].split('_')[1].split('.')[0])
        if pdb not in pdb_label_dic.keys():
            pdb_label_dic[pdb] = []
        i.append(rep_no)
        pdb_label_dic[pdb].append(i)


    for key in pdb_label_dic.keys():
        val = pdb_label_dic[key]
        val.sort(key = lambda x: x[-1])
        for i in val:
            i.pop()
        pdb_label_dic[key] = val
    return pdb_label_dic

def copynum_smoothening(preds_range, index, merge_cutoff):
    preds_len = len(preds_range)
    if preds_len <= 1 or preds_len <= index:
        return preds_range
    
    new_range = []
    repi0 = preds_range[index].split("-")[0]
    repi1 = preds_range[index].split("-")[1]

    # previous repeat
    repj0 = preds_range[index-1].split("-")[0]
    repj1 = preds_range[index-1].split("-")[1]

    if (int(repi0)-int(repj1)) <= merge_cutoff:
        for idx, i in enumerate(preds_range):
            if idx == index-1:
                continue
            elif idx == index:
                new_range.append(repj0+"-"+repi1)
            else:
                new_range.append(i)
        return copynum_smoothening(new_range, index, merge_cutoff)
    
    else:
        return copynum_smoothening(preds_range, index+1, merge_cutoff)

def smoothening_split(preds_range, bndry_len):
    new_range = []
    for i in preds_range:
        repi0 = i.split("-")[0]
        repi1 = i.split("-")[1]
        repbndry_len = int(repi1) - int(repi0) + 1
        
        mul_factor = round(float(repbndry_len)/bndry_len)
        if mul_factor > 1.0:
            new_blen = int(repbndry_len/mul_factor)
            bndry_res = [ str(j) for j in range(int(repi0), int(repi1)+1) ]
            cnt = 0
            startres = bndry_res[0]
            for j in bndry_res:
                if cnt < new_blen:
                    endres = j
                else:
                    new_range.append(startres+"-"+endres)
                    startres = j
                    cnt = 0
                cnt += 1

            new_range.append(startres+"-"+endres)
        else:
            new_range.append(i)
    return new_range

def remove_stray_residues(preds_range, cutoff_len):
    new_range = []
    for i in preds_range:
        repi0 = i.split("-")[0]
        repi1 = i.split("-")[1]
        repbndry_len = int(repi1) - int(repi0) + 1
        if repbndry_len > cutoff_len:
            new_range.append(i)
    return new_range

def bndry_residue_matching(preds_range, truth_data):
    truth_data_list = truth_data.split(";")
    truth_copynum = len(truth_data_list)
    midres_list = []
    for i in preds_range:
        repi0 = int(i.split("-")[0])
        repi1 = int(i.split("-")[1])
        midres = int((repi0+repi1)/2)
        midres_list.append(midres)

    truth_midres_list = []
    for idx, i in enumerate(truth_data_list):
        if idx < len(truth_data_list)-1:
            truth_midres_list.append(int(i.split("-")[1]))
    
    pred_len = len(midres_list)
    truth_len = len(truth_midres_list)
    if pred_len >= truth_len:
        return get_closest_residue_list_2(midres_list, truth_midres_list), abs(pred_len-truth_len)
    else:
        return get_closest_residue_list_2(truth_midres_list, midres_list), abs(pred_len-truth_len)
    
def get_closest_residue_list_2(pred_midres_list, truth_midres_list):
    pred_len = len(pred_midres_list)
    truth_len = len(truth_midres_list)
#     print (pred_midres_list)
#     print (truth_midres_list)
    closest_pairs = []
    if pred_len >= truth_len:
        i=0
        min_diffsum = 99999
        while i <= (pred_len-truth_len):
            diffsum = 0
            for j in range(i, i+truth_len):
                diffsum += abs(pred_midres_list[j]-truth_midres_list[j-i])
            if diffsum < min_diffsum:
                min_diffsum = diffsum
                min_diffsum_index = i
            i += 1
        
        for k in range(min_diffsum_index, min_diffsum_index+truth_len):
            closest_pairs.append((pred_midres_list[k], truth_midres_list[k-min_diffsum_index]))
        
#         print (closest_pairs, "\n\n")
    return closest_pairs

def ranges_repbndry(p, bndryres):
    q = sorted(p)
    i = 0
    for j in range(1,len(q)):
        if q[j] in bndryres:
            yield (str(q[i])+"-"+str(q[j-1]))
            i = j
    yield (str(q[i])+"-"+str(q[-1]))


# In[42]:


def print_bound_region(pdb_label_dic, merge_cutoff, bndry_len, cutoff_stray_len):
    
    f = open("results/residue_classification_ouput.txt", "w+")
    f.write("--------------- Deep-StRIP Prediction --------------------------\n")
    print("--------------- Deep-StRIP Prediction --------------------------\n")

    cnt = 0
    true_cn = []
    pred_cn = []
    true_midres = []
    pred_midres = []
    total_extra_residues = 0

    for i in pdb_label_dic:
        pred_val = []
        f.write("PDB Chain: "+ i+"\n")
        print ("PDB Chain: ", i)
        preds = []
        repeat_region = []
        for residue in pdb_label_dic[i]:
            repeat_name = residue[0].split(".")[0].upper()
            protein_name = repeat_name.split("_")[0]
            repeat_num = repeat_name.split("_")[1]
            pred_val = residue[1]
            if pred_val == 0:
                preds.append(int(repeat_num))
            if preds:
                preds_range = ranges(preds)

            repeat_region.append(int(repeat_num))


        if len(preds):
            preds_range = list(preds_range)

            preds_range = copynum_smoothening(preds_range, 1, merge_cutoff)
            preds_range = smoothening_split(preds_range, bndry_len)
            preds_range = remove_stray_residues(preds_range, cutoff_stray_len)
            closest_pairs, num_extra_residues = bndry_residue_matching(preds_range, pdb_repdata[i])
            total_extra_residues += num_extra_residues
            
            pred_cn.append(len(preds_range)+1)
            
            for k in closest_pairs:
                pred_midres.append(k[0])

        bndry_endpoints = []
        for s in preds_range:
            midres = (int(s.split("-")[0])+int(s.split("-")[1]))/2
            bndry_endpoints.append(int(midres))
        
        ans_range = ranges_repbndry(repeat_region, bndry_endpoints)
        f.write("Predicted Repeat Region: ")
        print("Predicted Repeat Region: ", end='')
        for s in ans_range:
            print (s+";", end='')
            f.write(s+";")

        
        f.write("\nPredicted copynumber: " + str(len(preds_range)+1))
        print("\nPredicted copynumber: " + str(len(preds_range)+1))
        f.write("\n---------------------------------------------------------\n")
        print("\n---------------------------------------------------------\n")
    
    f.close()
    
    return pred_cn, true_midres, pred_midres, total_extra_residues


# In[43]:


f = open("util_files/sequence_data.txt", "r")
pdb_repdata = {}
for line in f.readlines():
    l = line.strip().split("\t")
    pdb_repdata[(l[0]+l[1]).upper()] = l[-1]


# In[44]:


pdb_label_dic_cn = get_pdb_label_dic_cn(output_with_label_rb)
pdb_dic_cn = copy.deepcopy(pdb_label_dic_cn)
pred_cn, true_midres, pred_midres, total_extra_residues = print_bound_region(pdb_label_dic_cn, 0, 27, 9)


# In[ ]:




