# SVGRepresentation

## Datafolder

All code in SVGRepresentation expects the folder ‘SVG_Data’ to be in the same directory as the cloned project reposity.
```
parent/
    ├── SVG_Data
    └── SVG_LogoGenerator
```
If you have it in a different location make sure to adapt the `datafolder` variable inside scripts accordingly.

## How to install python requirements for executing code in SVGRepresentation

Create a venv with python 3.7
`python3 -m venv <path/to/new/virtual/environment>`

Activate it with
`</path/to/new/virtual/environment>/Scripts/activate`

Open the requirements.txt file in this folder (SVGRepresentation).
Check your cuda version and potentially adapt the following requirements to use your cuda version:  
`torch==1.11.0+cu115`
`torchvision==0.12.0+cu115`

Install dependencies (from inside the folder SVGRepresentation) with 
`pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html`



## For running the svg-preprocessing you additionally need requirements outside of python:
Download Inkscape 1.1.1 (3bf5ae0d25, 2021-09-20) from https://inkscape.org/release/inkscape-1.1.1/ .

Install Inkscape.

After installing Inkscape add the functionality of applying transforms by copying the files

 - `applytransform.inx`  
 - `applytransform.py`

from the `/SVGRepresentation/inkscape_fixed_apply_transforms_plugin` folder into your Inkscape directory under `<inkscape>/share/inkscape/extensions`.

Note that for running the [preprocess-svgs notebook](svg-preprocessing/preprocess-svgs.ipynb), you need to run on a machine with a GUI, because transform removal requires this.

## Structure of SVGRepresentation 

```
SVGRepresentation/
    ├── deepsvg
    ├── dense-represention
    ├── inkscape_fixed_apply_transforms_plugin
    ├── svg-preprocessing
    └── training
```
deepsvg 
- contains a clone of DeepSVG, with some adaptions and bugfixes
- in case you want to adapt training configuration files or create a new one, do so in `deepsvg/configs/deepsvg/<name_of_the_config>`
- additonally contains our stylistic Tensor representation in deepsvg/svg_tensor_with_styles - [README](deepsvg/svg_tensor_with_styles/README.md).

dense-representation
- contains code to inspect and save learned dense-representations 

inkscape_fixed_apply_transforms_plugin
- contains the Inkscape plugin files needed for transform removal in SVG files.

svg-preprocessing
- deals with deduplication, preprocessing and meta-file creation for SVG datafolders

training
- provides a notebook to train our DeepSVG models

## Training/Fine-Tuning DeepSVG models:

To aquire our latent SVG representations we trained a few DeepSVG models.
To train a selection of our DeepSVG models just follow the instructions in this [notebook](training/train_deepsvg_model.ipynb).

Have fun ! :)

