from .engine import GCNMultiLabelMAPEngine
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import train_test_split
from .util import Map
from .dataloader import LogoData
from .models import gcn_resnet101

class mlgcnTrain:
    def __init__(self, settingsDict, dataPath, labelEmbeddingPath, adjPath, checkpointPath):
        self.args = Map(settingsDict)
        self.dataPath = Path(dataPath)
        self.labelEmbeddingPath = Path(labelEmbeddingPath)
        self.adjPath = Path(adjPath)
        self.checkpointPath = Path(checkpointPath)

        self.dataPrepare()
        self.train()


    def dataPrepare(self):
        lld_labels = pd.read_csv(self.dataPath)
        lld_labels['cluster'] = lld_labels['Labels'].astype('category')
        lld_labels['labels'] = lld_labels['cluster'].cat.codes
        lld_labels['file_name'] = self.args.data + lld_labels['Name'].astype('str')
        self.lld_labels = lld_labels
        logo_data = lld_labels[['file_name', 'labels']]
        logo_data = logo_data.groupby(by='file_name')['labels'].apply(list).reset_index()
        logo_data = logo_data.to_dict(orient='records')

        train_logo, val_logo = train_test_split(logo_data, test_size=0.3, random_state=42)
        self.train_dataset = LogoData(train_logo, phase='train', logoData=logo_data, inp_name=str(self.labelEmbeddingPath))
        self.val_dataset = LogoData(
            val_logo, phase='val', logoData=logo_data, inp_name=str(self.labelEmbeddingPath))

    def train(self):
        use_gpu = torch.cuda.is_available()
        num_classes = self.lld_labels.labels.max() + 1


        # in channel defines length of vectorspace from bart_vec model
        model = gcn_resnet101(num_classes=num_classes, t=0.4,
                            adj_file=str(self.adjPath), in_channel=768)
        # define loss function (criterion)
        criterion = nn.MultiLabelSoftMarginLoss()

        # define optimizer
        optimizer = torch.optim.SGD(model.get_config_optim(self.args.lr, self.args.lrp),
                                    lr=self.args.lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay)

        state = {'batch_size': int(self.args.batch_size), 'image_size': int(self.args.image_size), 'max_epochs': int(self.args.epochs),
                'evaluate': int(self.args.evaluate), 'resume': self.args.resume, 'num_classes': num_classes}

        state['difficult_examples'] = True
        state['save_model_path'] = str(self.checkpointPath)
        state['workers'] = int(self.args.workers)
        state['epoch_step'] = int(self.args.epoch_step)
        state['lr'] = int(self.args.lr)
        # state['device_ids'] = args.device_ids
        if self.args.evaluate:
            state['evaluate'] = False
        engine = GCNMultiLabelMAPEngine(state)
        # engine.learning(model, criterion, self.train_dataset, self.val_dataset, optimizer)
