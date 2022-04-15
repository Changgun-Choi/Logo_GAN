# Conditional GAN model for font generation

# Dataset
The dataset contains alphabet PNG images avaiable at [kaggle](https://www.kaggle.com/datasets/thomasqazwsxedc/alphabet-characters-fonts-dataset). Downlaod and create folder 'A_Z_Images/images'. All the image folders will be present inside this.

# Model
Run CGAN.py

## Parameters
epochs:100  
image channels:1  
image size:32  
image shape: (channels, image size, image size)  
no of classes: 26  
latent_dim:100  
learning rate : 0.0002  
b1:0.5  
b2:0.999  
n_cpu:8  
sample nterval:400

# Output
Generated images while training can be found at 'results/'

# Old Results
Generated images for interval of batches are stored at 'oldresults/'

# Pre-trained Model
Model is saved under 'model/'
