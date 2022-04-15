
# import library
import torch
from torchvision import models, transforms
from PIL import Image


class classifyImage:
    def __init__(self, inputPath):
        self.imagePath = inputPath
        self.threshold = 0.5
        self.preprocess_img = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # normalize: has been experimentally shown that this helps the models to converge faster or achieve better result
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # imagenet
        ])
        # define labels and device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        with open(r"FeatureCreation/SupportingFiles/ImageLabels.txt", "r") as f:
            categories = [s.strip() for s in f.readlines()]
        # define model
        self.imageCategories = categories
        self.model = models.efficientnet_b7(pretrained=True)

    def preprocessImage(self):
        image = Image.open(self.imagePath).convert("RGB")
        image = self.preprocess_img(image).float()
        image = image.unsqueeze_(0)
        return image

    def classPrediction(self):
        preprocessedImage = self.preprocessImage()
        self.model.to(self.device)
        self.model.eval()
        input = preprocessedImage.to(self.device)
        output = self.model(input)
        probabilities = torch.nn.Softmax(dim=-1)(output)
        sortedProba = torch.argsort(probabilities, dim=-1, descending=True)
        objectProbability = {self.imageCategories[classIdx.item()]: probabilities[0, classIdx.item()].item() for (_, classIdx) in enumerate(sortedProba[0, :])}
        relevantPrediction = {category: probability for category, probability in objectProbability.items() if probability >= self.threshold}
        return relevantPrediction
