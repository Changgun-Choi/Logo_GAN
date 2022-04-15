import cv2
import pickle
from bbox import BBox2D
from tqdm import tqdm
import multiprocessing
from pathlib import Path
from multiprocessing.dummy import Pool as ThreadPool
from .Classes.ImageClassification import classifyImage
from .Classes.TextRecognition import textDetection
from .Classes.FaceRecognition import faceDetection


class ruleLogoLabelling:
    def __init__(self, pngFolderPath, textAreaThreshold, destPath):
        assert float(textAreaThreshold), 'Input for textAreaThreshold must be in format float'
        self.folderPath = Path(pngFolderPath)
        self.textAreaThreshold = textAreaThreshold
        self.destPath = Path(destPath)
        assert self.folderPath.is_dir(), 'pngFolderPath must be a directory'

    def logoDetection(self, logo):
        labelDict = {}
        logo = str(logo)
        areaThreshold = self.textAreaThreshold
        company = logo.split('.')[:-1]
        companyName = '.'.join(company)
        image = cv2.imread(logo)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 512))
        realEntityMatching = classifyImage(image)
        # start if image can recognize natural class
        classRecognition = realEntityMatching.classPrediction()
        if not classRecognition:
            pictorialMatch = False
        else:
            pictorialMatch = True
        # start text detection in image
        textRecognition = textDetection(image)
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
        facePrediction = faceDetection(image)
        faceDetected = facePrediction.detectFaces()
        if len(faceDetected) == 0:
            faceMatch = False
        else:
            faceMatch = True
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
        labelDict = {'Company': companyName, 'Filename': logo, 'Pictogram': pictorialMatch,
                    'Text': textMatch, 'Face': faceMatch, 'TextArea': area, 'category': logoCategory}
        return labelDict


    def ruleLabelling(self):
        listLogos = list(self.folderPath.glob('*.png'))
        resultsDict = [self.logoDetection(logo) for logo in tqdm(listLogos)]
        writeDict = open('FeatureCreation/labelDict.pickle', 'wb')
        pickle.dump(resultsDict, writeDict)



'''if __name__ == '__main__':
    # define method rules and parameters
    basePath = 'Data/WorldvectorPNG'
    listLogos = os.listdir(basePath)
    listLogos = [file for file in listLogos if file.endswith('.png')]
    # pool = ThreadPool(100)
    with multiprocessing.Pool(5) as pool:
        resultsDict = list(
            tqdm(pool.imap(logoDetection, listLogos), total=len(listLogos)))
    pool.close()
    pool.join()
    writeDict = open('FeatureCreation/labelDict.pickle', 'wb')
    pickle.dump(resultsDict, writeDict)'''
    
'''for _ in tqdm(pool.starmap(logoDetection, listLogosIterator), total=len(listLogos)):
        pass
    pool.close()
    pool.join()'''


'''# iterate over list of logos
for logo in tqdm(listLogos):
    iteration += 1
    if logo.endswith('.png'):
        company = logo.split('.')[:-1]
        companyName = '.'.join(company)
        # imagePath = f'{basePath}/{logo}'
        imagePath = f'{basePath}/{logo}'
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 512))
        realEntityMatching = classifyImage(image)
        # start if image can recognize natural class
        classRecognition = realEntityMatching.classPrediction()
        if not classRecognition:
            pictorialMatch = False
        else:
            pictorialMatch = True
        # start text detection in image
        textRecognition = textDetection(image)
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
        facePrediction = faceDetection(image)
        faceDetected = facePrediction.detectFaces()
        if len(faceDetected) == 0:
            faceMatch = False
        else:
            faceMatch = True
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
        labelDict[iteration] = {'Company': companyName, 'Filename': logo, 'Pictogram': pictorialMatch,
                                'Text': textMatch, 'Face': faceMatch, 'TextArea': area, 'category': logoCategory}
    else:
        continue
    if iteration % 100 == 0:
        writeDict = open('FeatureCreation/labelDict.pickle', 'wb')
        pickle.dump(labelDict, writeDict)
# save results into dataframe
logoLabel = pd.DataFrame.from_dict(labelDict, orient='index')
logoLabel.to_csv('Model/Label/Worldvector_ruleBasedLabels.csv', index=False)'''
