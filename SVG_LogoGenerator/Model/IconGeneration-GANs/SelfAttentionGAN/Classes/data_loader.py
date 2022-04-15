import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, image, transform_img=None):
        self.image = image
        self.transform_img = transform_img

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        image_loc = self.image[idx]
        image_data = Image.open(image_loc).convert('RGB')
        image_tensor = self.transform_img(image_data)
        return image_tensor


def create_data_loader(image_data, batch_size, transformer_img, num_workers=2, sampler=None):
    # initialize dataset by reference to image_data paths and image_target paths given in data df
    # -> apply transformation !
    ds = ImageDataset(
        image=np.array(image_data),
        transform_img=transformer_img)
    # initialize sampler based on given dataset -> here: RandomSampler
    if sampler is not None:
        sampler = sampler(ds)

    # return DataLoader object
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler
    )
