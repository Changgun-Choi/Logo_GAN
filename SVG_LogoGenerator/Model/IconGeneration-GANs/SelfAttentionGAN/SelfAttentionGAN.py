# import base packages
import os
from pathlib import Path
# import third-party packages
import pandas as pd
import numpy as np

# import torch packages
from torch.utils.data import RandomSampler
from torch.backends import cudnn
from torchvision import transforms
# import classes methods, classes
from Classes.data_loader import create_data_loader
from Classes.utils import make_folder, Map
from Classes.trainer import Trainer


def main(config, data):
    # import settings

    # For fast training
    cudnn.benchmark = True

    # preprocess image
    preprocess_img = transforms.Compose([
        transforms.CenterCrop(160),
        transforms.Resize(int(config.imsize)),
        transforms.ToTensor()
    ])
    # Data loader
    data_loader = create_data_loader(data.image, int(config.batch_size), preprocess_img, num_workers=config.num_workers, sampler=RandomSampler)
    # Create directories if not exist
    make_folder(config.model_save_path, config.version)
    make_folder(config.sample_path, config.version)
    make_folder(config.log_path, config.version)
    make_folder(config.attn_path, config.version)

    if config.train:
        if config.model == 'sagan':
            trainer = Trainer(data_loader, config)
        elif config.model == 'qgan':
            trainer = qgan_trainer(data_loader.loader(), config)
        trainer.train()
    else:
        tester = Tester(data_loader, config)
        tester.test()


if __name__ == '__main__':
    model_id = 1
    sagan_settings = pd.read_excel('Model\SelfAttentionGAN\Self Attention GAN Worksheet.xlsx')
    idx = sagan_settings[sagan_settings['version'] == f'sagan_{model_id}'].index[0] - 1
    config = Map(sagan_settings.iloc[idx].to_dict())
    # data path
    imageFolderPath = Path(config['image_path'])
    dataPath = os.listdir(imageFolderPath)
    iteration = 0
    dataDict = {}
    for path in dataPath:
        id = path.split('.')[0]
        imagePath = imageFolderPath / path
        dataDict[iteration] = {'id': id, 'image': imagePath}
        iteration += 1
    data = pd.DataFrame.from_dict(dataDict, orient='index')
    # model configuration

    print(config)
    ValueError('Test')
    main(config, data)
