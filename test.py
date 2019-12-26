import cv2
import pyttsx3
import math
import os


def TTS(name):
    engine = pyttsx3.init()
    engine.say(name)
    engine.runAndWait()



def recognize(img, clf, classifier,  num):
    
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, 1.1, 10)

    name = ""

    for (x, y, w, h) in features:
        label, conf = clf.predict(gray_img[y:y+h, x:x+w])
        print(conf)
        if(conf<35):
            for root, dirs, files in os.walk("Data/"):
                for d in dirs:
                    if d.startswith(str(label)):
                        name = str(d).partition(".")[2]
                        color = 255,250,250
                        break
        
        else: 
            name = "Unknown"
            color = 0,0,255

        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, name, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)  
    cv2.putText(img, "Number of Faces Detected: " + str(len(features)), (0,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)      

    if num % 10 == 0:
        TTS(name)

    return img


def main():
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.yml")

    video_capture=cv2.VideoCapture(0, cv2.CAP_DSHOW)

    num = 0

    while True:
        _, img = video_capture.read()
        img = recognize(img, clf, faceCascade, num)
        cv2.imshow("Face Detection", img)
        
        num = num + 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    