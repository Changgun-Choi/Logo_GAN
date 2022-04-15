# DeepVecFont model for font generation
The paper of this project is [available here](https://arxiv.org/pdf/2110.06688.pdf).

Implementation of a deepvecfont model based on the official source code (https://github.com/yizhiwang96/deepvecfont)


# Installation
Please follow the installation requirments mentioned [here](https://github.com/yizhiwang96/deepvecfont). 

# Dataset
This dataset is a subset from [SVG-VAE](https://github.com/magenta/magenta/tree/main/magenta/models/svg_vae)., ICCV 2019.  and it can be found [here](https://drive.google.com/drive/folders/1dGOOXK63-QJKXnE7_fD2OCfYJGKsApSg).
please download and put under ./data/dvf_main_model/

# Pre-Trained models
pre-trained models can be found under ./experiments/
The Image Super-resolution model Download links can be found [here](https://drive.google.com/drive/folders/1D_U4KHbt42u6ZGNNOAOvy5QXjwHj_abX).
Please download the image_sr dir and put it under ```  ./experiments/ ```

# Testing
The configurations can be found in options.py.

To test the main model, run
```
python test_sf.py --mode test --experiment_name dvf --model_name main_model --test_epoch 1500 --batch_size 1 --mix_temperature 0.0001 --gauss_temperature 0.01
```
The generated data will be stored in ./experiments/dvf_main_model/results/

# Font Merge

to merge characters to words run ./fontmerge.py script giving it two or three arguments, first, company name. Second, a 4-digit number ranging from (0000) to (1424). Third argument is optional, add png as the last argument if you want create a png version of the generated word
```
#example
python fontmerge.py Facebook 0001
```
Output:



![](output.svg)
