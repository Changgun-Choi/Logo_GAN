# import package
import cv2


class faceDetection:
    def __init__(self, inputPath):
        self.model = cv2.CascadeClassifier(r'FeatureCreation/ModelStates/haarcascade_frontalface_default.xml')
        self.imagePath = inputPath

    def detectFaces(self):
        img = cv2.imread(self.imagePath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = self.model.detectMultiScale(gray, 1.1, 4)
        return faces
