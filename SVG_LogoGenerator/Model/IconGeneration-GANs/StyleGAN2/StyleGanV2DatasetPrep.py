# import packages
import os
import pandas as pd
import json
from PIL import Image
from tqdm import tqdm
from pathlib import Path
# set script attributes
datasetPath = Path('Model/Label/LLD_majorityVote.csv')
destPath = Path('Data/StyleganV2MajorityVoteColab/')
destPath.mkdir(parents=True, exist_ok=True)
destPath = str(destPath)
datasetLabels = pd.read_csv(str(datasetPath))
datasetLabels
if datasetPath == Path('Model/Label/LLD_majorityVote.csv'):
    datasetLabels = datasetLabels[~datasetLabels['cluster'].isin(
        ['Underwater', 'Coffee cup'])].reset_index(drop=True)

datasetLabels
# transform dataframe to JSON structure required by STYLEGAN
datasetLabels['path'] = datasetLabels['Name']
datasetLabels['category'] = datasetLabels['cluster'].astype(
    'category').cat.codes
datasetLabels = datasetLabels[['path', 'category']]
listDatasetLabels = [[row['path'], row['category']]
                     for _, row in datasetLabels.iterrows()]
jsonData = {'labels': listDatasetLabels}
with open(destPath + str(Path('/')) + 'dataset.json', "w") as outfile:
    json.dump(jsonData, outfile)
print('JSON File created')
# Transform Greyscale image to RGB
sourcePath = 'Data/Conditional_StyleGAN_Logo'
imageList = os.listdir(sourcePath)
for _, row in tqdm(datasetLabels.iterrows(), 'Image', total=datasetLabels.shape[0]):
    imageName = row['path']
    image = Image.open('Data/Conditional_StyleGAN_Logo'
                       + str(Path('/')) + imageName).convert('RGB')
    image = image.resize((128, 128))
    image.save(destPath + str(Path('/')) + imageName)
print('Img File transformed')
