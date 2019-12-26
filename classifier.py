import numpy as np
from PIL import  Image
import os, cv2
import math

def train_classifier(data_dir):

    path = []
    for r, d, f in os.walk(data_dir):
        for file in f:
            if '.jpg' in file:
                path.append(os.path.join(r, file))


    faces = []
    labels = []
    
    for image in path:
        img = Image.open(image).convert('L')
        imageNp = np.array(img, 'uint8')
        label = int(str(image).partition("\\")[2].partition(".")[0])
        faces.append(imageNp)
        labels.append(label)

    labels = np.array(labels)
    clf = cv2.face.LBPHFaceRecognizer_create()
    
    clf.train(faces, labels)
    
    clf.write("classifier.yml")

if __name__ == "__main__":
    train_classifier("data")    