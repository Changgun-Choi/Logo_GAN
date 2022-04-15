import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import torch
import cv2
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
from cv2 import dnn_superres


class saliencyCombine:
    def __init__(self, imagePath, fontPath, destPath, position, plot):
        # load pretrained efficientnet b7 model
        self.model = torchvision.models.efficientnet_b7(pretrained=True)
        # define normalization to process input image
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        # define inv_normalize
        self.inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
            std=[1/0.229, 1/0.224, 1/0.255]
        )

        #normalize the image
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            self.normalize,
        ])

        self.imagePath = Path(imagePath)
        self.fontPath = Path(fontPath)
        self.destPath = Path(destPath)
        self.position = position
        self.plot = plot
        self.combination()

    def saliency(self, img, model, plot):
        #we don't need gradients w.r.t. weights for a trained model
        for param in model.parameters():
            param.requires_grad = False

        #set model in eval mode
        model.eval()
        #transoform input PIL image to torch.Tensor and normalize
        input = self.transform(img)
        input.unsqueeze_(0)

        #we want to calculate gradient of higest score w.r.t. input
        #so set requires_grad to True for input
        input.requires_grad = True
        #forward pass to calculate predictions
        preds = model(input)
        score, indices = torch.max(preds, 1)
        #backward pass to get gradients of score predicted class w.r.t. input image
        score.backward()
        #get max along channel axis
        slc, _ = torch.max(torch.abs(input.grad[0]), dim=0)
        #normalize to [0..1]
        slc = (slc - slc.min())/(slc.max()-slc.min())

        #apply inverse transform on image
        with torch.no_grad():
            input_img = self.inv_normalize(input[0])
        # calculate bounding box
        bboxArea = torch.where(slc > 0.3)
        # xmin, xmax, ymin, ymax for the bounding box
        ymin = torch.min(bboxArea[0]).item()
        ymax = torch.max(bboxArea[0]).item()
        xmin = torch.min(bboxArea[1]).item()
        xmax = torch.max(bboxArea[1]).item()
        width = xmax-xmin
        height = ymax-ymin
        bboxArea = (xmin, xmax, ymin, ymax)
        #plot image and its saleincy map
        if plot:
            plt.figure(figsize=(10, 10))
            plt.subplot(1, 2, 1)
            plt.imshow(np.transpose(input_img.detach().numpy(), (1, 2, 0)))
            plt.gca().add_patch(Rectangle((xmin, ymin), width, height,
                                          linewidth=1, edgecolor='r', facecolor='none'))
            plt.xticks([])
            plt.yticks([])
            plt.subplot(1, 2, 2)
            plt.imshow(slc.numpy(), cmap=plt.cm.hot)
            plt.gca().add_patch(Rectangle((xmin, ymin), width, height,
                                          linewidth=1, edgecolor='r', facecolor='none'))
            plt.xticks([])
            plt.yticks([])
            plt.show()
        return slc, bboxArea

    def resizeImage(self, size):

        sr = dnn_superres.DnnSuperResImpl_create()

        # Read image
        image = cv2.imread(str(self.imagePath))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Read the desired model
        sr.readModel('ImageCombination/LapSRN_x8.pb')

        # Set the desired model and scale to get correct pre- and post-processing
        sr.setModel("lapsrn", size)
        # Upscale the image
        result = sr.upsample(image)
        result = Image.fromarray(result)
        return result

    def combination(self):
        img = self.resizeImage(8)
        font = Image.open(str(self.fontPath)).convert('RGBA')
        font = font.crop(font.getbbox())
        slc, bboxArea = self.saliency(img, self.model, self.plot)
        img = img.resize((224, 224))
        width = bboxArea[1] - bboxArea[0]
        height = bboxArea[3] - bboxArea[2]
        font.thumbnail((width, height), Image.ANTIALIAS)
        if self.position == 'overlay':
            xposition = int(bboxArea[0])
            yposition = int(bboxArea[2] + ((bboxArea[3] - bboxArea[2]) / 2))
            img.paste(font, (xposition, yposition), mask=font)
        elif self.position == 'below':
            xposition = int(bboxArea[0])
            yposition = max(
                int((img.size[1] - bboxArea[2] - (font.size[1] / 2))), 0)
            img.paste(font, (xposition, yposition), mask=font)
        elif self.position == 'above':
            xposition = int(bboxArea[0])
            yposition = max(int(bboxArea[2] - (font.size[1] / 2)), 0)
            img.paste(font, (xposition, yposition), mask=font)
        img.save(str(self.destPath))
