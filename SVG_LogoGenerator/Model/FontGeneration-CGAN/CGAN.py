#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

import os
from PIL import Image
from torch.utils.data import Dataset
#DEVICE = 'cpu'
device = "cuda" if torch.cuda.is_available() else "cpu" 
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("GeneratedClassImages", exist_ok=True)


# In[ ]:


n_epochs=int(100)
channels=int(1)
img_size=int(32)
img_shape = (channels, img_size, img_size)
n_classes=26
latent_dim=int(100)
lr=0.0002
b1=0.5
b2=0.999
n_cpu=int(8)
sample_interval=int(400)
cuda = True if torch.cuda.is_available() else False


# In[ ]:


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(n_classes, n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img


# In[ ]:


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(n_classes, n_classes)

        self.model = nn.Sequential(
            nn.Linear(n_classes + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity


# In[ ]:


# Loss functions
adversarial_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

word2index = {"A":0, "B":1, "C":2,"D":3,"E":4,"F":5,"G":6,"H":7,"I":8,"J":9,"K":10,"L":11,"M":12,"N":13,"O":14,"P":15,"Q":16,"R":17,"S":18,"T":19,"U":20,"V":21,"W":22,"X":23,"Y":24,"Z":25}

def word_2_index(category,word2index):
    word_2_index = word2index[category]
    return word_2_index

class StyleDataset_edit(Dataset):
    def __init__(self, data_dir=None, transform=None, label_dir=None, word2index=None):
        
        folders=[os.path.join('A_Z_Images/Images', x) for x in ('A','B','C','D','E','F','G' ,'H','I','J','K','L','M','N','O', 'P','Q','R','S','T','U','V', 'W','X','Y','Z')]
        
        
        iteration = 0
        dataDict = {}
        filename=[]
        labels_list=[]
        for paths in folders:
            dataPath = os.listdir(paths)
            for path in dataPath:
                id = path.split('.')[0]
                imagePath = f'./{paths}/{path}'                                            
                dataDict[iteration] = {'id': id, 'image': imagePath, 'label': paths[-1]}
                iteration += 1
        #data = pd.DataFrame.from_dict(dataDict)
        for items in dataDict.values():
            filename.append(items['image']) 
            labels_list.append(items['label'])                                                    
        self.full_filenames =  filename  
        #labels_list = [labels for labels in data['label']]
                                                                
        
        self.labels = [word_2_index(category, word2index) for category in labels_list]
        self.transform = transform
        

    def __len__(self):
        # return size of dataset
        return len(self.full_filenames)

    def __getitem__(self, idx):
        # open image, apply transforms and returna with label
        image = Image.open(self.full_filenames[idx])  # Full path to open image
  

        #image = image.convert('RGB')
        image = self.transform(image)  
        
        return image, self.labels[idx]

# define a simple transformation that only converts a PIL image into PyTorch tensors
import torchvision.transforms as transforms
data_transformer = transforms.Compose([
    transforms.Resize(32),  
    transforms.ToTensor(),  
    transforms.CenterCrop(32),  ### necessary
    transforms.Normalize(mean=(0.5),std=(0.5)) # -1 ~ 1

])


# In[ ]:


word2index = {"A":0, "B":1, "C":2,"D":3,"E":4,"F":5,"G":6,"H":7,"I":8,"J":9,"K":10,"L":11,"M":12,"N":13,"O":14,"P":15,"Q":16,"R":17,"S":18,"T":19,"U":20,"V":21,"W":22,"X":23,"Y":24,"Z":25}

font_dataset = StyleDataset_edit(None,data_transformer,None,word2index)

dataloader = torch.utils.data.DataLoader(font_dataset, batch_size=128, shuffle=True, num_workers=0, drop_last=True)


# In[ ]:


# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, "results/%d.png" % batches_done, nrow=n_row, normalize=True)


# In[ ]:


for epoch in range(n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, n_classes, batch_size)))

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator(real_imgs, labels)
        d_real_loss = adversarial_loss(validity_real, valid)

        # Loss for fake images
        validity_fake = discriminator(gen_imgs.detach(), gen_labels)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % sample_interval == 0:
            sample_image(n_row=26, batches_done=batches_done)


# In[ ]:


#%%  
path2models = 'model/'
os.makedirs(path2models, exist_ok=True)
path2weights_gen = os.path.join(path2models, 'con_weights_gen_new.pt')
path2weights_dis = os.path.join(path2models, 'con_weights_dis_new.pt')
torch.save(generator.state_dict(), path2weights_gen)
torch.save(discriminator.state_dict(), path2weights_dis)


# In[ ]:


# use cuda if available
DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else "cpu")

# Load
weights = torch.load(path2weights_gen)
generator.load_state_dict(weights)

# evalutaion mode
generator.eval()


# In[ ]:


# Load
weights = torch.load(path2weights_dis)
discriminator.load_state_dict(weights)

# evalutaion mode
discriminator.eval()


# In[ ]:


#%%  Plot generated image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
def generate_plot(samples):
        fig = plt.figure(figsize = (12,12))
        gs = gridspec.GridSpec(12,12)
        gs.update(wspace = 0.05, hspace = 0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(32,32), cmap = 'gray')
        return fig 


# In[ ]:


# fake image 
with torch.no_grad():
    z = Variable(FloatTensor(np.random.normal(0, 1, (128, 100))))
    gen_labels = Variable(LongTensor(np.random.randint(0, 26, 128)))    
    img_fake = generator(z, gen_labels).detach().cpu() 

    
print(img_fake.shape)
print(z.shape)
print(gen_labels.shape)


# In[ ]:


temp=np.zeros(128)
temp[:]=1
print(gen_labels)
print(Variable(LongTensor(temp)))


# In[ ]:


def create(inp):  # 26 characters
      with torch.no_grad():
            z = Variable(FloatTensor(np.random.normal(0, 1, (128, 100))))
            temp=np.zeros(128)
            temp[:]=int(inp)
            
            # run the traineg generator excluding Discriminator

            generated_samples = generator(z, Variable(LongTensor(temp))).detach().cpu() 
            save_image(generated_samples.data, "GeneratedClassImages/%d.png" % inp, nrow=12, normalize=True)

            generate_plot(generated_samples)


# In[ ]:


for keys in word2index.keys():
    create(word2index[keys])

