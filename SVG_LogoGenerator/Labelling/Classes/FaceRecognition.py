# import package
import cv2


class faceDetection:
    def __init__(self, image):
        self.model = cv2.CascadeClassifier(
            r'FeatureCreation/ModelStates/haarcascade_frontalface_default.xml')
        self.image = image

    def detectFaces(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = self.model.detectMultiScale(gray, 1.1, 4)
        return faces
