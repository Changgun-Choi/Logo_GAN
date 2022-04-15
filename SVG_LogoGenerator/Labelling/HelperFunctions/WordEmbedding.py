from transformers import BartTokenizer, BartModel
from sentence_transformers import SentenceTransformer
import pandas as pd
import pickle
from pathlib import Path


class wordEmbedding:
    def __init__(self, dataPath, pretrainedPath, filePath):
        self.dataPath = Path(dataPath)
        self.pretrainedPath = pretrainedPath
        self.filePath = Path(filePath)

    def sentenceTransformer(self):
        # read data
        data = pd.read_csv(self.dataPath)
        uniqueLabel = data['Labels'].unique().tolist()
        # load model
        model = SentenceTransformer(self.pretrainedPath)
        # initiate list of numpy arrays
        labelEmbedding = model.encode(uniqueLabel)
        with open(self.filePath, 'wb') as f:
            pickle.dump(labelEmbedding, f)
        print('Word embedding successfully transformed')
