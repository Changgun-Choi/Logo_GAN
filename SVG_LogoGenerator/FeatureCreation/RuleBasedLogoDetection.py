import os
import cv2
import pandas as pd
from bbox import BBox2D
from Classes.ImageClassification import classifyImage
from Classes.TextRecognition import textDetection
from Classes.FaceRecognition import faceDetection
# define method rules and parameters
basePath = 'Data/Conditional_StyleGAN_Logo'
listLogos = os.listdir(basePath)
iteration = 0
areaThreshold = 0.55
labelDict = {}
# iterate over list of logos
for logo in listLogos:
    if logo.startswith('.'):
        continue
    else:
        company = logo.split('.')[:-1]
        companyName = '.'.join(company)
        # imagePath = f'{basePath}/{logo}'
        imagePath = f'{basePath}/{logo}'
        image = cv2.imread(imagePath)
        realEntityMatching = classifyImage(imagePath)
        # start if image can recognize natural class
        classRecognition = realEntityMatching.classPrediction()
        if not classRecognition:
            pictorialMatch = False
        else:
            pictorialMatch = True
        # start text detection in image
        textRecognition = textDetection(f'{basePath}/{logo}')
        textDetected = textRecognition.eastDetect()
        # calculate area of text coverage in png
        area = 0
        for (startX, startY, endX, endY) in textDetected:
            box = BBox2D([startX, startY, endX, endY], mode=1)
            area += (box.h * box.w) / (320*320)
        if len(textDetected) == 0:
            textMatch = False
        else:
            textMatch = True
        # start face recognition in image
        facePrediction = faceDetection(imagePath)
        faceDetected = facePrediction.detectFaces()
        if len(faceDetected) == 0:
            faceMatch = False
        else:
            faceMatch = True
        iteration += 1
        # match logo categories based on rule
        if pictorialMatch is False and textMatch is True and faceMatch is False and area >= areaThreshold:
            logoCategory = 'Wordmark'
        elif pictorialMatch is True and textMatch is False and faceMatch is False:
            logoCategory = 'Pictorial mark'
        elif pictorialMatch is False and textMatch is False and faceMatch is False:
            logoCategory = 'Abstract mark'
        elif textMatch is True or area < areaThreshold and faceMatch is False:
            logoCategory = 'Combination mark'
        elif faceMatch is True:
            logoCategory = 'Mascot logo'
        else:
            logoCategory = 'Emblem logo'
        labelDict[iteration] = {'Company': companyName, 'Pictogram': pictorialMatch,
                                'Text': textMatch, 'Face': faceMatch, 'TextArea': area, 'category': logoCategory}
        print(f'{iteration} from {len(listLogos)}')
# save results into dataframe
logoLabel = pd.DataFrame.from_dict(labelDict, orient='index')
logoLabel.to_csv('Model/Label/LLD_ruleBasedLabels.csv', index=False)
