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
# https://wikidocs.net/52460


#%%
# Use GPU if CUDA is available
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
print(torch.__version__)

device = "cuda" if torch.cuda.is_available() else "cpu" 
torch.manual_seed(1)

#%%
# define an object of the custom dataset for the train folder
data_dir=  'C:/Users/ChangGun Choi/Team Project/TeamProject/SVG_Logo/Data/Conditional_StyleGAN_Logo/'
label_dir = "C:/Users/ChangGun Choi/Team Project/TeamProject/SVG_Logo/Model/Label/LLD_weightedMajorityVote.csv"
word2index = {"Serveware":0, "Art":1, "Vehicle":2, 'T-shirt': 3, 'Terrestrial animal':4, 'Plastic':5, 'Coffee cup': 6, 'Musician': 7, 'Underwater':8,  'Machine gun': 9 }
labels_df = pd.read_csv(label_dir,sep = ",", header = None, skiprows=1 )
Logo_dataset = StyleDataset(data_dir,data_transformer, label_dir ,word2index)
dataloader = torch.utils.data.DataLoader(Logo_dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
dataloader   # Randomly sample batch, drop_last=True : length of Dataset is not dividable by the batch size without a remainder, which happens to be 1
#%%
label = [labels for (imgs, labels) in dataloader]
label[0]
#print(n)
#print(imgs.shape())  # input image     # torch.Size([4, 3, 28, 28]) (batch_size, RGB, pixel, pixel)
 # print(labels)    #tensor([1, 1]) : batch is 2 
len(dataloader)
#%%
def weights_init(m):
  classname = m.__class__.__name__

  if classname.find('Conv') != -1:
    nn.init.normal_(m.weight.data, 0.0,0.02)
  
  elif classname.find('BatchNorm') != -1:
    nn.init.normal_(m.weight.data,1.0,0.02)
    nn.init.constant_(m.bias.data,0)
    
#%%
params = {'nz':100, # noise 
          'ngf':128, # generator conv filter 
          'ndf':128, # discriminator conv filter 
          'img_channel':3, 
          'num_classes': 10              ######################### Change
          }

class Generator(nn.Module):
    def __init__(self, params):
        super().__init__()
        nz = params['nz'] # noise  100
        ngf = params['ngf'] # conv filter 
        img_channel = params['img_channel'] 
        num_class = params['num_classes']
        self.relu= nn.LeakyReLU(0.2)
        self.dconv1 = nn.ConvTranspose2d(nz,ngf*4, 4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(ngf*4)  # (ngf*8)*4*4
        self.dconv1_l = nn.ConvTranspose2d(num_class, ngf*4, 4, stride=1, padding=0, bias=False)
        self.bn1_l = nn.BatchNorm2d(ngf*4)  # (ngf*8)*4*4
        
        self.dconv2 = nn.ConvTranspose2d(ngf*8,ngf*4, 4, stride=2, padding=1, bias=False) # ngf*8: dimension
        self.bn2 = nn.BatchNorm2d(ngf*4) # (ngf*4)*8*8
        self.dconv3 = nn.ConvTranspose2d(ngf*4,ngf*2,4,stride=2,padding=1,bias=False)
        self.bn3 = nn.BatchNorm2d(ngf*2)  # (ngf*2)*16*16
        self.dconv4 = nn.ConvTranspose2d(ngf*2,ngf,4,stride=2,padding=1,bias=False)
        self.bn4 = nn.BatchNorm2d(ngf)    # ngf * 32 * 32
        self.dconv5 = nn.ConvTranspose2d(ngf,img_channel,4,stride=2,padding=1,bias=False)
                                          # img_channel * 64 *64
    def forward(self,noise,label):
        x = self.relu(self.bn1(self.dconv1(noise)))
        y = self.relu(self.bn1_l(self.dconv1_l(label)))
        #x = self.bn1(self.dconv1(noise))
        #y = self.bn1_l(self.dconv1_l(label))
       
        x = torch.cat([x,y], 1)                  
        
        x = self.relu(self.bn2(self.dconv2(x)))
        x = self.relu(self.bn3(self.dconv3(x)))
        x = self.relu(self.bn4(self.dconv4(x)))
        x = torch.tanh(self.dconv5(x))
        return x
#%%

image_size = 64
label_dim = 10   ### 3
onehot_dim = torch.LongTensor([i for i in range(label_dim)])

# label preprocess
onehot = torch.zeros(label_dim, label_dim)
onehot = onehot.scatter_(1, onehot_dim.view(label_dim, 1), 1).view(label_dim, label_dim, 1, 1) 
onehot
fill = torch.zeros([label_dim, label_dim, image_size, image_size])
for i in range(label_dim):
    fill[i, i, :, :] = 1     # image_size * image_size -> make 1 

# check
x = torch.randn(image_size,100,1,1, device=device).view(-1, 100, 1, 1).cuda() # batch,100,1,1
label = torch.randint(0,label_dim,(image_size,), device=device)  # 3
label   
label = onehot[label].to(device)
label
model_gen = Generator(params).to(device)
#model_gen = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()  # if you have more GPUs
#model_gen = model_gen.to(f'cuda:0')

out_gen = model_gen(x,label)  # 3*64*64  
print(out_gen.shape)      #torch.Size([64, 3, 64, 64])

#%% Ex1)
class Discriminator(nn.Module):
    def __init__(self,params):
        super().__init__()
        img_channel = params['img_channel'] # 3
        ndf = params['ndf'] # 128
        num_class = params['num_classes']
        self.relu= nn.LeakyReLU(0.2)

        self.conv1 = nn.Conv2d(img_channel,ndf//2,4,stride=2,padding=1,bias=False)
        self.conv1_l = nn.Conv2d(num_class,ndf//2,4,stride=2,padding=1,bias=False)
        
        self.conv2 = nn.Conv2d(ndf,ndf*2,4,stride=2,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(ndf*2)
        self.conv3 = nn.Conv2d(ndf*2,ndf*4,4,stride=2,padding=1,bias=False)
        self.bn3 = nn.BatchNorm2d(ndf*4)
        self.conv4 = nn.Conv2d(ndf*4,ndf*8,4,stride=2,padding=1,bias=False)
        self.bn4 = nn.BatchNorm2d(ndf*8)
        self.conv5 = nn.Conv2d(ndf*8,    1,4,stride=1,padding=0,bias=False)  # discriminate 1 or 0

    def forward(self, img, label):                    # input : img_channel * 64 * 64
        x = self.relu(self.conv1(img))                # ndf * 32 * 32
        y = self.relu(self.conv1_l(label)) 
        x = torch.cat([x,y],1)
        
        x = self.relu(self.bn2(self.conv2(x))) #(ndf*2) * 16 *16
        x = self.relu(self.bn3(self.conv3(x))) #(ndf*4)*8*8
        x = self.relu(self.bn4(self.conv4(x))) #(ndf*8)*4*4
        x = torch.sigmoid(self.conv5(x))              # 1*1*1
        return x.view(-1,1).squeeze(0)         # dimension: 1 -> Probability
                                               # sequeeze() : delete size 1  =>> Discriminator wants one value
#%%
x = torch.randn(image_size,params['img_channel'],64,64,device=device) # num = 16
#x = torch.randn(image_size,100,1,1, device=device).view(-1, 100, 1, 1).cuda() # batch,100,1,1
label = torch.randint(0,label_dim,(image_size,), device=device)  # 3
label = fill[label].to(device)
label
model_dis = Discriminator(params).to(device)
out_dis = model_dis(x, label)
print(out_dis)

#%%
torch.cuda.is_available()
# Initialization
generator = Generator(params).to(device)    
discriminator = Discriminator(params).to(device)
generator.apply(weights_init).cuda
discriminator.apply(weights_init).cuda

# Loss Function
adversarial_loss = nn.BCELoss()
adversarial_loss.cuda()

# learning rate
lr = 0.0002 #2e-4

# Optimzation
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))


# Visualize
#fixed_noise = torch.randn(4, 100, 1, 1, device=DEVICE)
#%%

from PIL import Image 
Image.MAX_IMAGE_PIXELS = 1000000000 
import cv2

#https://stackoverflow.com/questions/21596281/how-does-one-convert-a-grayscale-image-to-rgb-in-opencv-python
#https://pypi.org/project/opencv-python/

print(torch.version)
import torchvision.utils as vutils 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from IPython.display import HTML

real_batch = next(iter(dataloader)) 
plt.figure(figsize=(10,10)) 
plt.axis("off") 
plt.title("Training Images") 
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
len(dataloader)
#%%
from torch.autograd import Variable
def tensor2var(x, grad=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=grad)

#%%
image_size = 64
label_dim = 10


params = {'nz':100, # 
          'ngf':128, 
          'ndf':128, 
          'img_channel':3,
          'num_classes': 10              ##### Change
          }

n_epochs =70 # How many epochs
img_list = []
iters = 0
sample_interval = 100 # Which interval of Batch, want to see the results
#start_time = time.time()
loss_history={'gen':[],
              'dis':[]}

fixed_z = torch.normal(mean=0, std=1, size=(image_size, 100)).view(-1, 100, 1, 1).cuda()  # (input_d , latent_dim)
fixed_label = torch.randint(0,label_dim,(image_size,)).to(device) # random label  0~9  
fixed_label = onehot[fixed_label].to(device)
fixed_z

#%%
model_save_step = len(dataloader)
model_save_step

for epoch in range(n_epochs):
    for i, (imgs, label) in enumerate(dataloader):
        generator.train()
        discriminator.train()
        
        #imgs = cv2.cvtColor(img, cv.CV_GRAY2RGB)  #####
        # Real image        real_imgs = imgs.to(device)
        real_labels = label.to(device) #################  
        # Creating Ground_truth Discriminator_label
        real = torch.ones(imgs.size(0), 1).to(device)# real: 1   # imgs.size(0) : Batch Size 
        fake = torch.zeros(imgs.size(0), 1).to(device) # fake: 0

        """ Train generator """
        
        optimizer_G.zero_grad()
        # Sampling random noise

        noise = torch.normal(mean=0, std=1, size=(imgs.shape[0], 100)).view(-1, 100, 1, 1).cuda()  # (input_d , latent_dim)
        noise
        label = torch.randint(0,label_dim,(imgs.size(0),)).to(device) # random label  0~9  
        
        gen_label = onehot[label].to(device)
        gen_label
        dis_label = fill[label].to(device)
        # Generate fake image
        generated_imgs = generator(noise, gen_label) #######
        # Discriminate fake image
        out_dis = discriminator(generated_imgs, dis_label)
        
        # Loss calculate
        g_loss = adversarial_loss(out_dis, real) # real : 1 
                                #[0.6,0.7,....]
        # generator update
        g_loss.backward()
        optimizer_G.step()
        ###########################################################################
        
        """ Train discriminator """
        optimizer_D.zero_grad()
        
        # Real image
        real_dis_labels = fill[real_labels].to(device)
        out_dis = discriminator(real_imgs, real_dis_labels)
        real_loss = adversarial_loss(out_dis, real)
        
        # Fake Image
        out_dis = discriminator(generated_imgs.detach(),dis_label)
        fake_loss = adversarial_loss(out_dis, fake) # not update G() during training D()
        # Detaching fake from the graph is necessary to avoid forward-passing the noise through G when we actually update the generator. 
        # If we do not detach, then, although fake is not needed for gradient update of D, it will still be added to the computational graph
        # and as a consequence of backward pass which clears all the variables in the graph (retain_graph=False by default), fake won't be available when G is updated.
        d_loss = (real_loss + fake_loss) / 2

        # update
        d_loss.backward()
        optimizer_D.step()
        
        loss_history['gen'].append(g_loss.item())
        loss_history['dis'].append(d_loss.item())
        
        path = 'C:/Users/ChangGun Choi/Team Project/TeamProject/data/Generated image/con_DCGAN/eliminated/new/'
        path2models = 'C:/Users/ChangGun Choi/Team Project/TeamProject/model/Conditional_GAN/model_save/'
        
        done = epoch * len(dataloader) + i
        # Sample images 
        if (done) % sample_interval == 0:
            fake_images = generator(fixed_z, fixed_label) 
            save_image(denorm(fake_images.data),
                       os.path.join(path, '{}_fake.png'.format(done)))  

    #if (i+1) % model_save_step==0:
    torch.save(generator.state_dict(),
               os.path.join(path2models, '{}_G.pth'.format(epoch)))
    torch.save(discriminator.state_dict(),
               os.path.join(path2models, '{}_D.pth'.format(epoch)))   
    
    # print log each epoch
    print(f"[Epoch {epoch}/{n_epochs}] [D loss: {d_loss.item():.6f}] [G loss: {g_loss.item():.6f}]")
    
#%%
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.title('Loss Progress')
plt.plot(loss_history['gen'], label='Gen. Loss')
plt.plot(loss_history['dis'], label='Dis. Loss')
plt.xlabel('batch count')
plt.ylabel('Loss')
plt.legend()
plt.show()

#%%  

path2models = 'C:/Users/ChangGun Choi/Team Project/TeamProject/model/Conditional_GAN/'
os.makedirs(path2models, exist_ok=True)
path2weights_gen = os.path.join(path2models, 'weights_gen.pt')   #TeamProject/model/self_attention/93960_D.pth
path2weights_dis = os.path.join(path2models, 'weights_dis.pt')  # TeamProject/model/self_attention/93960_G.pth

torch.save(generator.state_dict(), path2weights_gen)
torch.save(discriminator.state_dict(), path2weights_dis)
#%%
# Load
weights = torch.load("TeamProject/model/self_attention/93960_G.pth")
generator.load_state_dict(weights)

# evalutaion mode
model_gen.eval()

#%%
# fake image 
with torch.no_grad():
    
    fixed_noise = torch.randn(64,100,1,1, device=device).view(-1, 100, 1, 1).cuda() # batch,100,1,1
    label = torch.randint(0,3,(64,), device=device)   # Category 2#################
    gen_label = onehot[label].to(device)
    gen_label    
    img_fake = model_gen(fixed_noise, gen_label).detach().cpu() 
   
print(img_fake.shape)

#%% Create labels to fit into Coditional GAN

def create(inp):  
  with torch.no_grad():
      feature_map = {"Serveware":0, "Art":1, "Vehicle":2, 'T-shirt': 3, 'Terrestrial animal':4, 'Plastic':5, 'Coffee cup': 6, 'Musician': 7, 'Underwater':8,  'Machine gun': 9 }
      Y_dimension = 10
      samples = 64
      z_noise = torch.randn(64,100,1,1).view(-1, 100, 1, 1).to(device) # batch,100,1,1
      
      
      #Y_label = torch.zeros(0,3,(64,), device=device) 
      Y_label = torch.full((64, ), feature_map[inp]).to(device)
      Y_label = onehot[Y_label].to(device)
      # run the traineg generator excluding Discriminator
      
      generated_samples = model_gen(z_noise, Y_label).detach().cpu()
      output = generated_samples
      output.shape[0]
      for j in range(output.shape[0]):
            id =  j
            output_ = output[j]
            plt.imshow(np.transpose(output_,(1,2,0)))    
create("Serveware")
create("Vehicle")
create("Art")


