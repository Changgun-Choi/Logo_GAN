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

    def bartTokenizer(self):
        data = pd.read_csv(self.dataPath)
        uniqueLabel = data['Labels'].unique().tolist()
        tokenizer = BartTokenizer.from_pretrained(self.pretrainedPath)
        model = BartModel.from_pretrained(self.pretrainedPath)
        inputs = tokenizer(uniqueLabel, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        labelVectors = last_hidden_states.cpu().detach().numpy()
        labelVectors = labelVectors.reshape(len(uniqueLabel), -1)
        with open(self.filePath, 'wb') as f:
            pickle.dump(labelVectors, f)
        print(f'Word embedding successfully transformed. Unique Labels:{len(uniqueLabel)} \n Shape Tensor: {labelVectors.shape}')

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
