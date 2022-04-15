import pickle
import numpy as np
import torch.utils.data as data
from PIL import Image


class LogoData(data.Dataset):
    def __init__(self, image_label, logoData, transform=None, phase='train', inp_name=None):
        self.phase = phase
        self.image_label = image_label
        self.transform = transform
        self.logoData = logoData
        self.num_classes = len(
            set([label for element in self.logoData for label in element['labels']]))
        

        with open(inp_name, 'rb') as f:
            self.inp = pickle.load(f)
        self.inp_name = inp_name

    def __len__(self):
        return len(self.image_label)

    def __getitem__(self, index):
        item = self.image_label[index]
        return self.get(item)

    def get(self, item):
        filename = item['file_name']
        labels = sorted(item['labels'])
        img = Image.open(filename).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        target = np.zeros(self.num_classes, np.float32) - 1
        target[labels] = 1
        return (img, filename, self.inp), target
