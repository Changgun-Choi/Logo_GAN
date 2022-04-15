# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 22:55:04 2021

@author: ChangGun Choi

"""
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
# fix torch random seed
torch.manual_seed(0)

def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


#%% one_hot_encoding labels creation

def one_hot_encoding(word2index, labels):
  one_hot_label = []
  for category in word2index:  
      one_hot_vector = [0]*(len(word2index))
      index = word2index[category]
      
      if labels == index:
          one_hot_vector[index] = 1
          one_hot_label.append(one_hot_vector)
  return one_hot_label


#%%

def word_2_index(category,word2index):
    word_2_index = word2index[category]
    return word_2_index

 

def one_hot_label_create(category,word2index):
      one_hot_vector = [0]*(len(word2index))
      index = word2index[category]
      one_hot_vector[index] = 1
      #one_hot_label.append(one_hot_vector)
      return one_hot_vector

#one_hot_label_create("Pictorial", word2index)

#%%

class StyleDataset_edit(Dataset):
    def __init__(self, data_dir, transform, label_dir, word2index):
     
        #data_dir ='C:/Users/ChangGun Choi/Team Project/TeamProject/SVG_Logo/Data/Conditional_StyleGAN_Logo'
        #label_dir = "C:/Users/ChangGun Choi/Team Project/TeamProject/SVG_Logo/Model/Label/LLD_weighted_eliminated.csv"
        path2data = data_dir

        # get a list of images
        filenames = os.listdir(path2data)  # png name_list
        filenames = [f[:-4] for f in filenames] #[:-4] .png format delete
        filenames[-2]
        
        path2csvLabels = label_dir   
        labels_df = pd.read_csv(path2csvLabels,sep = ",",header = None, skiprows=1 )
        labels_df.rename(columns = {0: "Company", 1: "label"}, inplace=True)
        labels_df[labels_df["label"] == 'Camera lens']
        labels_company =  [labels[:-4] for labels in labels_df['Company']]
        labels_list = [labels for labels in labels_df['label']]
        # get the full path to images
        filenames = [f for f in filenames for l in labels_company if f==l]
       
        self.full_filenames = [os.path.join(path2data, f+ '.jpg') for f in filenames]  # Path              
        # = [labels_df.loc[labels].values[6] for labels in labels_name] #[:-4] .png delete
        #labels = [word_2_index(category, word2index) for category in labels_list] 
        self.labels = [word_2_index(category, word2index) for category in labels_list]
        self.transform = transform
        
    def __len__(self):
        # return size of dataset
        return len(self.full_filenames)

    def __getitem__(self, idx):
        # open image, apply transforms and return with label
        image = Image.open(self.full_filenames[idx])  # Full path to open image
        #Image.open(full_filenames[0])  # Full path to open image

        image = image.convert('RGB')
        image = self.transform(image)  
        
        return image, self.labels[idx]

# define a simple transformation that only converts a PIL image into PyTorch tensors
import torchvision.transforms as transforms
data_transformer = transforms.Compose([
    transforms.Resize(64),  
    transforms.ToTensor(),  
    transforms.CenterCrop(64),  ### necessary
    transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)) # -1 ~ 1
    #transforms.Normalize(mean=(0.5),std=(0.5)) # -1 ~ 1

])

#%%
class StyleDataset(Dataset):
    def __init__(self, data_dir, transform, label_dir, word2index):
        path2data = data_dir

        # get a list of images
        filenames = os.listdir(path2data)  # png name_list
        #filenames = [f[:-4] for f in filenames] #[:-4] .png format delete
        filenames[0]
        
         # labels are in a csv file named train_labels.c
        path2csvLabels = label_dir   
        labels_df = pd.read_csv(path2csvLabels,sep = ",",header = None, skiprows=1 )
        labels_df.rename(columns = {0: "Company", 1: "label"}, inplace=True)
        labels_list = [labels for labels in labels_df['label']]
        labels_list[:10]
       
        self.full_filenames = [os.path.join(path2data, f ) for f in filenames]  # Path    
       
        self.labels = [word_2_index(category, word2index) for category in labels_list]
        #labels = [one_hot_label_create(category, word2index) for category in labels_list]
        self.transform = transform
        

    def __len__(self):
        # return size of dataset
        return len(self.full_filenames)

    def __getitem__(self, idx):
        # open image, apply transforms and return with label
        image = Image.open(self.full_filenames[idx])  # Full path to open image
        #Image.open(full_filenames[0])  # Full path to open image

        image = image.convert('RGB')
        image = self.transform(image)  
        
        return image, self.labels[idx]

# define a simple transformation that only converts a PIL image into PyTorch tensors
import torchvision.transforms as transforms
data_transformer = transforms.Compose([
    transforms.Resize(64),  
    transforms.ToTensor(),  
    transforms.CenterCrop(64),  ### necessary
    transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)) # -1 ~ 1
])

#%%
class LogoDataset(Dataset):
    def __init__(self, data_dir, transform, data_type='train'):
        path2data = os.path.join(data_dir, data_type)

        # get a list of images
        filenames = os.listdir(path2data)  # png name_list
        filenames = [f[:-4] for f in filenames] #[:-4] .png format delete
        filenames[0]
        
         # labels are in a csv file named train_labels.csv
        csv_filename = data_type + '_labels.csv'
        path2csvLabels = os.path.join(data_dir, csv_filename)      
        labels_df = pd.read_csv(path2csvLabels,sep = ",",header = None, skiprows=1 )
        labels_df.rename(columns = {1: "Company", 7: "label"}, inplace=True)

        labels_name = labels_df['Company'].to_list()

        # get the full path to images
        filenames = [f for f in filenames for l in labels_name if f==l]  # FILE = LABELS
        self.full_filenames = [os.path.join(path2data, f + '.png') for f in filenames]  # Path
      
        # set data frame index to id
        labels_df.set_index('Company', inplace=True)
        # obtain labels from data frame
        self.labels = [labels_df.loc[labels].values[6] for labels in filenames] 
        self.transform = transform

    def __len__(self):
        # return size of dataset
        return len(self.full_filenames)

    def __getitem__(self, idx):
        # open image, apply transforms and return with label
        image = Image.open(self.full_filenames[idx])  # only selected png files open
        image = image.convert('RGB')
        image = self.transform(image)
        
        return image, self.labels[idx]

# define a simple transformation that only converts a PIL image into PyTorch tensors
import torchvision.transforms as transforms
data_transformer = transforms.Compose([
    transforms.Resize(64),  
    transforms.ToTensor(),  
    transforms.CenterCrop(64),  ### necessary
    transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)) # -1 ~ 1
  ])

#%%
class Nolabel_Dataset(Dataset):
    def __init__(self, data_dir, transform):
        path2data = os.path.join(data_dir)
        # get a list of images
        filenames = os.listdir(path2data)  # png name_list
        
        self.full_filenames = [os.path.join(path2data, f) for f in filenames]  # Path
        self.transform = transform

    def __len__(self):
        # return size of dataset
        return len(self.full_filenames)

    def __getitem__(self, idx):
        # open image, apply transforms and return with label
        image = Image.open(self.full_filenames[idx])  # only selected png files open
        image = image.convert('RGB')
        image = self.transform(image)
        
        return image

# define a simple transformation that only converts a PIL image into PyTorch tensors
import torchvision.transforms as transforms
data_transformer = transforms.Compose([
    transforms.Resize(64),  
    transforms.ToTensor(),  
    transforms.CenterCrop(64),  ### necessary
    transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)) # -1 ~ 1
    #transforms.Normalize(mean=(0.5),std=(0.5)) # -1 ~ 1

])

