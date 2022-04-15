# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 17:04:01 2021

@author: ChangGun Choi
"""

# Use GPU if CUDA is available
#DATA_PATH =  "C:/Users/ChangGun Choi/Team Project/png_file"
#MODEL_PATH = "C:/Users/ChangGun Choi/Team Project/model"

import os
import sys
import torch
import torch.nn as nn

from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
from help import *
#from util import nextplot   
from PIL import Image 
from matplotlib.pyplot import imshow

torch.cuda.is_available()  
#DEVICE = 'cpu'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 

#%%
transforms_train = transforms.Compose([
    transforms.Resize(64),  
    transforms.ToTensor(),  # 0 ~ 255
    transforms.CenterCrop(64), 
    transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)) # -1 ~ 1 normalized
])
#path = "C:/Users/ChangGun Choi/Desktop/0. 수업자료/3_videos_project/TEAM Project/SVG_LogoGenerator/Data"
path=  'C:/Users/ChangGun Choi/Team Project/TeamProject/SVG_Logo/Data/Conditional_StyleGAN_Logo/'

train_dataset = datasets.ImageFolder(root=path, transform=transforms_train)
Wdataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
                                                                    # Randomly sample batch
# drop_last : length of Dataset is not dividable by the batch size without a remainder, which happens to be 1                                                              
train_dataset.__getitem__(0)                                        
len(train_dataset)
#%%


    # Load data.
images, labels = zip(*[train_dataset[i] for i,  in enumerate(train_dataset)])
#%%
for i, data in enumerate(dataloader):
    print(data[0].size())  # input image     # torch.Size([4, 3, 28, 28]) (batch_size, RGB, pixel, pixel)
    #print(data[1])         # class label
    
#%%

latent_dim = 100      # dimension for latent vector "Z"
  
class Generator(nn.Module):                                    
    def __init__(self):
        super(Generator, self).__init__()

        # Defining 1 Block
        def block(input_dim, output_dim, normalize=True):
            layers = [nn.Linear(input_dim, output_dim)]        
            if normalize:                                      # Perceptron 
                # batch normalization -> Same dimension
                layers.append(nn.BatchNorm1d(output_dim, 0.8)) 
            layers.append(nn.LeakyReLU(0.2, inplace=True))     # Non-linear 
            return layers

        # Many blocks are sequentially stacked        # MLP perceptron
        self.model = nn.Sequential(                   # each block
            *block(latent_dim, 128, normalize=False), # (Z dimension, output_dim) 
            *block(128, 256),                 
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, 3 * 64 * 64),            # Fully-Connteced Flatten:(64 * 64) 
            nn.Tanh()                                # input [-1, 1] normalizing, generator use tanh for output
            )       
    def forward(self, z):
        img = self.model(z)                          # Generater_model: latent Z input    
      
        img = img.view(img.size(0), 3, 64, 64)       # Create image after applying Tanh 
        return img                                   
    
#%%
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(3 * 64 * 64, 512),   # Flattened input from : image 3 RGB * (64*64) 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),             # Linear 
            nn.Sigmoid(),                  # classification 0 ~ 1
        )

    # Give back result of discrimation
    def forward(self, img):
        flattened = img.view(img.size(0), -1)  # (64, 3 * 28 * 28)  Flattened input
        output = self.model(flattened)     
 
        return output
    
#%%
def weights_init(m):
  classname = m.__class__.__name__

  if classname.find('Conv') != -1:
    nn.init.normal_(m.weight.data, 0.0,0.02)
  
  elif classname.find('BatchNorm') != -1:
    nn.init.normal_(m.weight.data,1.0,0.02)
    nn.init.constant_(m.bias.data,0)
    
#%%
# Initialization
generator = Generator().to(DEVICE)
discriminator = Discriminator().to(DEVICE)
generator.apply(weights_init).cuda
discriminator.apply(weights_init).cuda

generator.cuda()
discriminator.cuda()

# Loss Function
adversarial_loss = nn.BCELoss()
adversarial_loss.cuda()

# learning rate
lr = 0.0002

# Optimzation
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Visualize Purpose
fixed_noise = torch.randn(4, 100, 1, 1, device=DEVICE)
#%%
n_epochs = 300 # How many epochs
sample_interval = 1000 # Which interval of Batch, want to see the results
start_time = time.time()

for epoch in range(n_epochs):
    for i, (imgs, label) in enumerate(dataloader):

        #imgs = cv2.cvtColor(img, cv.CV_GRAY2RGB)  #####
        # Real image
        real_imgs = imgs.cuda()   
        # Creating Ground_truth Label 
        real = torch.cuda.FloatTensor(imgs.size(0), 1).fill_(1.0) # real: 1   # imgs.size(0) : Batch Size 
        fake = torch.cuda.FloatTensor(imgs.size(0), 1).fill_(0.0) # fake: 0

        """ Train generator """
        optimizer_G.zero_grad()

        # Sampling random noise
        z = torch.normal(mean=0, std=1, size=(imgs.shape[0], latent_dim)).cuda()  # (input_d , latent_dim)
                                                                                 
        # Generate image
        generated_imgs = generator(z)

        # Loss calculate
        g_loss = adversarial_loss(discriminator(generated_imgs), real)  

        # generator update
        g_loss.backward()
        optimizer_G.step()
        ###########################################################################
        
        """ Train discriminator """
        optimizer_D.zero_grad()

        # discriminator loss
        real_loss = adversarial_loss(discriminator(real_imgs), real)
        fake_loss = adversarial_loss(discriminator(generated_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        # update
        d_loss.backward()
        optimizer_D.step()

        done = epoch * len(dataloader) + i
        if done % sample_interval == 0:  
            # Show only 4 images from generated pics
            save_image(generated_imgs.data[:4], f"{done}.png", nrow=2, normalize=True)
           #nrows: column, #normalize: If True, shift the image to the range (0, 1)
           #https://aigong.tistory.com/183
    # print log each epoch
    print(f"[Epoch {epoch}/{n_epochs}] [D loss: {d_loss.item():.6f}] [G loss: {g_loss.item():.6f}] [Elapsed time: {time.time() - start_time:.2f}s]")

#%%

# example of calculating the frechet inception distance
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm

# calculate frechet inception distance
def calculate_fid(act1, act2):
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = numpy.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid

# define two collections of activations
act1 = random(10*2048)
act1 = act1.reshape((10,2048))
act2 = random(10*2048)
act2 = act2.reshape((10,2048))
# fid between act1 and act1
fid = calculate_fid(act1, act1)
print('FID (same): %.3f' % fid)
# fid between act1 and act2
fid = calculate_fid(act1, act2)
print('FID (different): %.3f' % fid)