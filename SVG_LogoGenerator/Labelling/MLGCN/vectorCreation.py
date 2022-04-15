import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from pathlib import Path
from sklearn.cluster import KMeans
from tqdm import tqdm
from .dataloader import LogoData
from .models import gcn_resnet101
from .util import Warp, Map

class mlgcnVector:
    def __init__(self, settingsDict, modelCheckpointPath, dataPath, adjPath, destPath):
        self.args = Map(settingsDict)
        self.modelCheckpointPath = Path(modelCheckpointPath)
        self.dataPath = Path(dataPath)
        self.adjPath = Path(adjPath)
        self.destPath = Path(destPath)
        
        self.vectorization()

    def vectorization(self):
        lld_labels = pd.read_csv(self.dataPath)
        lld_labels['cluster'] = lld_labels['Labels'].astype('category')
        lld_labels['labels'] = lld_labels['cluster'].cat.codes
        lld_labels['file_name'] = self.args.data + lld_labels['Name'].astype('str')

        logo_data = lld_labels[['file_name', 'labels']]
        logo_data = logo_data.groupby(by='file_name')[
            'labels'].apply(list).reset_index()
        logo_data = logo_data.to_dict(orient='records')

        num_classes = lld_labels.labels.max() + 1
        model = gcn_resnet101(num_classes=num_classes,
                              t=0.4, adj_file=str(self.adjPath), in_channel=768)
        checkpoint = torch.load(str(self.modelCheckpointPath))
        model.load_state_dict(checkpoint['state_dict'])

        state = {'batch_size': int(self.args.batch_size), 'image_size': int(self.args.image_size), 'max_epochs': int(self.args.epochs),
                 'evaluate': int(self.args.evaluate), 'resume': self.args.resume, 'num_classes': num_classes, 'workers': int(self.args.workers)}
        
        normalize = transforms.Normalize(mean=model.image_normalization_mean,
                                         std=model.image_normalization_std)


        state['val_transform'] = transforms.Compose([
            Warp(state['image_size']),
            transforms.ToTensor(),
            normalize,
        ])
        data_set = LogoData(logo_data, phase='val',
                            inp_name='logo_label_embedding.pkl')
        # define train and val transform
        data_set.transform = state['val_transform']
        # data loading code
        output_data_loader = torch.utils.data.DataLoader(data_set,
                                                        batch_size=state['batch_size'], shuffle=True,
                                                        num_workers=state['workers'])

        imageName = tuple()
        outputTensorList = []
        model.eval().cuda()
        for i, (input, target) in enumerate(tqdm(output_data_loader)):
            feature = input[0]
            outName = input[1]
            imageName = imageName + outName
            vectorInput = input[2]

            feature = torch.autograd.Variable(feature).float().cuda()
            input = torch.autograd.Variable(vectorInput).float().cuda().detach()
            outputVector = model(feature, input)
            outputVector = outputVector.cpu().detach().numpy()
            outputTensorList.append(outputVector)
        
        outputTensor = np.concatenate(outputTensorList, axis=0)
        vectorDict = dict(zip(imageName, outputTensor))
        clusterVector = list(vectorDict.values())
        clusterArray = np.array(clusterVector)
        kmeans = KMeans(n_clusters=10, random_state=14).fit(clusterArray)
        kmeansLables = kmeans.labels_
        imageName = pd.unique(lld_labels.Name)
        imageName = list(imageName)
        imageLabels = tuple(zip(imageName, kmeans.labels_))
        clusterLabel = pd.DataFrame.from_records(imageLabels)
        clusterLabel.to_csv(self.destPath, index=False)
