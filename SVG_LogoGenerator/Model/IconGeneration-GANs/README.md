# GANs

The goal is to train a GAN to generate diverse Logos regarding categories (labels). The implementations of GAN models are built in pytorch w/w.o pretrained parameters. The challenges of training GAN include the difficulty in balancing generator and discriminator and mode collapse (over-fitting). Therefore, different models and techniques are implemented to compare results and solve the problems. 

## Table of Contents
- [Dataset](#dataset)
- [Main Model: StyleGAN 2](#stylegan-2)
  - [Requirements](#requirements)
  - [Data Preparation](#data-preparation)
  - [Python Code Editor](#python-code-editors)
  - [Alternative: Google Colab](#colab)
- [DCGAN Models](#dcgan-models)
- [Self-attention GAN](#self-attention-gan)

## Dataset
We use Boosted [LLD-logo dataset](https://data.vision.ee.ethz.ch/sagea/lld/#paper) from ConditionalStyleGAN github repository5 which removed
all text-based images and extended the remaining logos with image based logos and illustrations
scraped off of Google images.
#### Data available [here](https://drive.google.com/open?id=1ruFmYOc4q3D9aQOXg8RPdYPnBUcoa_iY)

## StyleGAN 2
https://github.com/NVlabs/stylegan2-ada-pytorch
### Requirements
- conda install -c anaconda python=3.7
- conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
  - CUDA toolkit 11.0 or later. Use at least version 11.1 if running on RTX 3090
- pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3
- git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git
- git clone https://github.com/Miriamgoldilocks/SVG_LogoGenerator.git

### Data Preparation
PNG files and a metadata file dataset.json for labels created from `Model/IconGeneration-GANs/StyleGAN2/StyleGanV2DatasetPrep.py` 


### Python Code Editors

#### 1. Create Json file (dictionary of logos and labels)
   - Run [StyleGanV2DatasetPrep.py](StyleGAN2/StyleGanV2DatasetPrep.py)
   - Output: 'dataset.json'
   - Locate the above json file with logos in `SVG_LogoGenerator/Data/StyleganV2MajorityVoteColab/`

#### 2. Create StyleGAN dataset
- Input: StyleGAN dataset containing Metadata file dataset.json for labels
- Resolution 32x32
#### Our final model is trained on Majority Vote (Methods to create labels)
Majority_Votes_Labels(Main Model)
* For training on Majority Vote:
```sh
python dataset_tool.py --source SVG_LogoGenerator/Data/StyleganV2MajorityVoteColab/ --dest SVG_LogoGenerator/Data/majority_vote --width=32 --height=32
```
ML_GCN_Labels
* For training on MLGCN:
```sh
python dataset_tool.py --source SVG_LogoGenerator/Data/StyleganV2MLGCNColab/ --dest SVG_LogoGenerator/Data/gcn --width=32 --height=32
```
* The two respective datasets after running `dataset_tool.py` are provided here:
[Majority Vote](https://drive.google.com/drive/folders/15Z3p-s-IRxng3e2jN54YGKURQNbiymDp?usp=sharing) /
[MLGCN](https://drive.google.com/drive/folders/1-A02RISoApQx6jWf38PbOznJ8tfD-TWY?usp=sharing)

#### 3. Train model 
#### Transfer-learning from pretrained-Cifar10 
Majority_Votes_Labels(Main Model)
```sh
python train.py --data SVG_LogoGenerator/Data/data_majority_new --outdir SVG_LogoGenerator/results --cond=1 --cfg=auto --mirror=1 --aug=ada --augpipe=bg --target=0.6 --gpus=1 --resume=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl
```
   
ML_GCN_Labels
```sh
python train.py --data SVG_LogoGenerator/Data/gcn --outdir SVG_LogoGenerator/results --cond=1 --cfg=auto --mirror=1 --aug=ada --augpipe=bg --target=0.6 --gpus=1 --resume=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl
```
#### 4. Generate Images
Majority_Votes_Labels(8 classes)
The following link contains our last trained networks which can be used to generate images:
* [Majority Vote (pretrained with Cifar10)](https://drive.google.com/file/d/1DkW3-kuSaZBKJBW0w8DvQ83T3ToQYDkN/view?usp=sharing)
- Generate class 1 as an example
- network: assign the location for downloaded pickles file
```sh
python generate.py --class=1 --network=StyleGAN/results/00037-data-cond-auto1/network-snapshot-001400.pkl --outdir=StyleGAN/results/generated --trunc=1 --seeds=30-48
```

### Alternative: Google Colab <a name="colab"></a>
To train it alternatively on an adapted Colab version from [Jeff Heaton](https://www.youtube.com/channel/UCR1-GEpyOPzT2AO4D_eifdw), please look at [this notebook](https://drive.google.com/file/d/1jV9Y5ZcvYlsjDO701EraUTk3ALYpX_1S/view?usp=sharing).

To access our datasets on Google Drive that are ready for training, please refer to the following links:
#### Our final model is trained on Majority Vote (Methods to create labels)
* For training on [Majority Vote](https://drive.google.com/drive/folders/15Z3p-s-IRxng3e2jN54YGKURQNbiymDp?usp=sharing)
* For training on [MLGCN](https://drive.google.com/drive/folders/1-A02RISoApQx6jWf38PbOznJ8tfD-TWY?usp=sharing)

The following links contain our last trained networks which can be used to generate images:

* Resume from [Majority Vote (pretrained with Cifar10)](https://drive.google.com/file/d/1DkW3-kuSaZBKJBW0w8DvQ83T3ToQYDkN/view?usp=sharing)
* Resume from [MLGCN (pretrained with Cifar10)](https://drive.google.com/file/d/1DhbkWuCmahXxpsGOuibnlquGzI_hcuuQ/view?usp=sharing)

Both networks use below as the best hyperparameter:
```
--cond=1 --cfg=auto --mirror=1 --aug=ada --augpipe=bg --target=0.6 --gpus=1
```



## DCGAN Models

- GAN MLP
- DCGAN
- Con_DCGAN (3 labels, 10 labels)


### Import help.py for using Pytorch dataloader with our logo dataset and labels

  1) Load png(logos) and label(csv file) at the same time and match them (Same dataset with StyleGAN2)
   #### Data available [here](https://drive.google.com/open?id=1ruFmYOc4q3D9aQOXg8RPdYPnBUcoa_iY)
  
  2) 'word_2_index' function returns label encoding for each label  
    ex. word2index = {"Serveware":0, "Art":1, "Vehicle":2, 'T-shirt': 3, 'Terrestrial animal':4, 'Plastic':5, 'Coffee cup': 6, 'Musician': 7, 'Underwater':8,  'Machine gun': 9 }
   - Possible to generate new logos with labels by typing input "Art"
  
  3) Returns Tuple (image, label)

### Input format: (Logo_image, labels)  
    imgs.shape() = (batch_size, RGB, pixel, pixel) 
    ex. torch.Size([64, 3, 28, 28])
   - Output: Generated logos from 'fixed noise' with Loss history
   - Details and explainations of each line of code is included as annotation (ex. # Generater_model )

### Models Run by Python Notebook
    DCGANs\Con_DCGAN_Server.ipynb
      
## Self-attention GAN

