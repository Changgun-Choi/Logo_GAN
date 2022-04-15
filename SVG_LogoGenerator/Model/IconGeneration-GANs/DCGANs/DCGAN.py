# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 09:32:58 2021

@author: ChangGun Choi
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 17:04:01 2021

@author: ChangGun Choi
"""
#%%
# Use GPU if CUDA is available
import os
import sys
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import os
import time
import numpy as np

from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
from help import *     # help.py created
#from util import nextplot   
from PIL import Image 
from matplotlib.pyplot import imshow
torch.cuda.is_available()  
device = "cuda" if torch.cuda.is_available() else "cpu" 
#cd "C:/Users/ChangGun Choi/Team Project/TeamProject"
pwd
# Set random seed for reproducibility
manualSeed = 2020
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed);
#img = Image.open('C:/Users/ChangGun Choi/Team Project/png_file/7-eleven.png') 
#imshow(np.asarray(img))


#%%

# define an object of the custom dataset for the train folder
data_dir=  'C:/Users/ChangGun Choi/Team Project/TeamProject/SVG_Logo/Data/Conditional_StyleGAN_Logo/'
Logo_dataset = StyleGAN_Dataset(data_dir, data_transformer)
dataloader = torch.utils.data.DataLoader(Logo_dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
dataloader   # Randomly sample batch, drop_last=True : length of Dataset is not dividable by the batch size without a remainder, which happens to be 1
len(dataloader)

#print(imgs.shape())  # input image     # torch.Size([64, 3, 28, 28]) (batch_size, RGB, pixel, pixel)

#%%  GAN https://wikidocs.net/62306 ,  https://dreamgonfly.github.io/blog/gan-explained/
# Generator Input : Latent vector z (batch_size x 100), Output: batch_size x 3*64*64 RGB
params = {'nz':100, # noise 
          'ngf':64, # generator conv filter 
          'ndf':64, # discriminator conv filter 
          'img_channel':3, # 
          }

class Generator(nn.Module):
    def __init__(self, params):
        super().__init__()
        nz = params['nz'] # 
        ngf = params['ngf'] # 
        img_channel = params['img_channel']

        self.dconv1 = nn.ConvTranspose2d(nz,ngf*8,4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(ngf*8)
        self.dconv2 = nn.ConvTranspose2d(ngf*8,ngf*4, 4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ngf*4)
        self.dconv3 = nn.ConvTranspose2d(ngf*4,ngf*2,4,stride=2,padding=1,bias=False)
        self.bn3 = nn.BatchNorm2d(ngf*2)
        self.dconv4 = nn.ConvTranspose2d(ngf*2,ngf,4,stride=2,padding=1,bias=False)
        self.bn4 = nn.BatchNorm2d(ngf)
        self.dconv5 = nn.ConvTranspose2d(ngf,img_channel,4,stride=2,padding=1,bias=False)

    def forward(self,x):
        x = F.relu(self.bn1(self.dconv1(x)))
        x = F.relu(self.bn2(self.dconv2(x)))
        x = F.relu(self.bn3(self.dconv3(x)))
        x = F.relu(self.bn4(self.dconv4(x)))
        x = torch.tanh(self.dconv5(x))
        return x

# check
x = torch.randn(1,100,1,1, device=device)
model_gen = Generator(params).to(device)
out_gen = model_gen(x)
print(out_gen.shape)
#%%
# Discriminator: 
class Discriminator(nn.Module):
    def __init__(self,params):
        super().__init__()
        img_channel = params['img_channel'] # 3
        ndf = params['ndf'] # 64

        self.conv1 = nn.Conv2d(img_channel,ndf,4,stride=2,padding=1,bias=False)
        self.conv2 = nn.Conv2d(ndf,ndf*2,4,stride=2,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(ndf*2)
        self.conv3 = nn.Conv2d(ndf*2,ndf*4,4,stride=2,padding=1,bias=False)
        self.bn3 = nn.BatchNorm2d(ndf*4)
        self.conv4 = nn.Conv2d(ndf*4,ndf*8,4,stride=2,padding=1,bias=False)
        self.bn4 = nn.BatchNorm2d(ndf*8)
        self.conv5 = nn.Conv2d(ndf*8,1,4,stride=1,padding=0,bias=False)

    def forward(self,x):
        x = F.leaky_relu(self.conv1(x),0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)),0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)),0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)),0.2)
        x = torch.sigmoid(self.conv5(x))
        return x.view(-1,1)

# check
x = torch.randn(16,3,64,64,device=device)
model_dis = Discriminator(params).to(device)
out_dis = model_dis(x)
print(out_dis.shape)
#%%
# 
def initialize_weights(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)

model_gen.apply(initialize_weights);
model_dis.apply(initialize_weights);

#%%
# Initialization
#generator = Generator(params)
#discriminator = Discriminator(params)

# 손실 함수 정의
loss_func = nn.BCELoss()

# 최적화 함수
from torch import optim
lr = 2e-4
beta1 = 0.5
beta2 = 0.999

opt_dis = optim.Adam(model_dis.parameters(),lr=lr,betas=(beta1,beta2))
opt_gen = optim.Adam(model_gen.parameters(),lr=lr,betas=(beta1,beta2))
#fixed_noise = torch.randn(4, 100, 1, 1, device=DEVICE)
len(dataloader)
#%%
fixed_noise = torch.randn(64, 100, 1, 1, device=device)
path = 'data/Generated image/DCGAN/new/'

sample_interval = 100 # Which interval of Batch, want to see the results
model_gen.train()
model_dis.train()

batch_count=0
num_epochs=100
#start_time = time.time()
nz = params['nz'] 
loss_hist = {'dis':[],
             'gen':[]}

for epoch in range(num_epochs):
    for i, (xb,label) in enumerate(dataloader):
        #print(xb.shape)
        #print(label.shape)
        
        ba_si = xb.shape[0]

        xb = xb.to(device)
        yb_real = torch.Tensor(ba_si,1).fill_(1.0).to(device)
        yb_fake = torch.Tensor(ba_si,1).fill_(0.0).to(device)

        # generator
        model_gen.zero_grad()

        z = torch.randn(ba_si,nz,1,1).to(device) # noise
        out_gen = model_gen(z) 
        out_dis = model_dis(out_gen) 

        g_loss = loss_func(out_dis,yb_real)
        g_loss.backward()
        opt_gen.step()

        # discriminator
        model_dis.zero_grad()
        
        out_dis = model_dis(xb) 
        loss_real = loss_func(out_dis,yb_real)

        out_dis = model_dis(out_gen.detach()) 
        loss_fake = loss_func(out_dis,yb_fake)

        d_loss = (loss_real + loss_fake) / 2
        d_loss.backward()
        opt_dis.step()

        loss_hist['gen'].append(g_loss.item())
        loss_hist['dis'].append(d_loss.item())

               
        done = epoch * len(dataloader) + i
        if done % sample_interval == 0:  
            fake_images = model_gen(fixed_noise)
            save_image(fake_images, os.path.join(path, f"{done}.png"), normalize=True)
            #os.path.join(path, f"{done}.png') 
           #nrows: column, #normalize: If True, shift the image to the range (0, 1)
           #https://aigong.tistory.com/183
    # print log each epoch
    print(f"[Epoch {epoch}/{num_epochs}] [D loss: {d_loss.item():.6f}] [G loss: {g_loss.item():.6f}]")
    
#%%
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.title('Loss Progress')
plt.plot(loss_hist['gen'], label='Gen. Loss')
plt.plot(loss_hist['dis'], label='Dis. Loss')
plt.xlabel('batch count')
plt.ylabel('Loss')
plt.legend()
plt.show()

#%%  
path2models = 'C:/Users/ChangGun Choi/Team Project/model/'
os.makedirs(path2models, exist_ok=True)
path2weights_gen = os.path.join(path2models, 'DCGAN_weights_gen.pt')
path2weights_dis = os.path.join(path2models, 'DCGAN_weights_dis.pt')

torch.save(model_gen.state_dict(), path2weights_gen)
torch.save(model_dis.state_dict(), path2weights_dis)

# Load
weights = torch.load(path2weights_gen)
model_gen.load_state_dict(weights)

# evalutaion mode
model_gen.eval()
# fake image 

with torch.no_grad():
    fixed_noise = torch.randn(16, 100,1,1,device=device)
    #label = torch.randint(0,2,(16,), device=device)   # Category 2#################
    img_fake = model_gen(fixed_noise).detach().cpu() 
print(img_fake.shape)
