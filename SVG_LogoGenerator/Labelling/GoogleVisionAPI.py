import io
import os
import cv2
import glob
import pandas as pd
from pathlib import Path
from google.cloud import vision
from tqdm import tqdm

class visionApi:
    def __init__(self, credentialsPath, imagePath, destPath):
        self.credentialsPath = credentialsPath
        self.imagePath = Path(imagePath)
        self.destPath = Path(destPath)

    def api(self):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] =self.credentialsPath
        ext = ['png', 'jpg', 'gif']    # Add image formats here
        files = []
        [files.extend(glob.glob(str(self.imagePath) + '*.' + e)) for e in ext]
        name = []
        piclabels = []
        scores = []

        for f in tqdm(files):
            name.append(f[61:])
            #print(f[61:])
            # Instantiates a client
            client = vision.ImageAnnotatorClient()

            # Loads the image into memory
            with io.open(f, 'rb') as image_file:
                content = image_file.read()

            image = vision.Image(content=content)

            # Performs label detection on the image file
            response = client.label_detection(image=image)
            labels = response.label_annotations
            #print(response)
            temp = []
            tempscore = []
            for label in labels:
                temp.append(label.description)
                tempscore.append(label.score)
                #print(label.description,':',label.score)
            piclabels.append(list(temp))
            scores.append(list(tempscore))

        df = pd.DataFrame(list(zip(name, piclabels)),
                        columns=['Name', 'Labels'])
        df['Scores'] = scores
        dfObj = pd.DataFrame(columns=['Names', 'Labels', 'Scores'])
        dfObj = df.explode('Labels')
        dfObj['Scores'] = df['Scores'].explode()
        dfObj.to_csv(str(self.destPath), index=False)
