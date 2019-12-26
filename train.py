import cv2
import os
import base64
import glob
import classifier



def data_entry():
    
    name = input("Enter your name: ")
   
    flag = 0
        
    for root, dirs, files in os.walk("Data/"):
        for d in dirs:
            if d.endswith(name):
                flag = 1
                break

    if flag == 0:

        if len(os.listdir("Data/")) == 0:
            u_id = 1   
        else:
            u_id = int(max([os.path.join("Data/",d) for d in os.listdir("Data/")], key=os.path.getmtime).partition("/")[2].partition(".")[0]) + 1
            
        os.mkdir(os.path.join("Data/", str(u_id) + ". " + name))

        print("User Added!")

    else:
        print("User already Exists!")
        u_id = int(d.partition(".")[0])

    return name, u_id



def generate_dataset(img, u_id, name, img_id):
    cv2.imwrite("Data/" + str(u_id) + ". " + name + "/" + str(img_id) + ".jpg", img)



def draw_boundary(img, classifier, scaleFactor, minNeighbours, color, text):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbours)
    coords = []

    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, text, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = (x, y, w, h)

    return coords



def detect(img, faceCascade, u_id, name, img_id):
    color = {"blue": (255,0,0), "red":(0,0,255), "green":(0,255,0)}
    coords = draw_boundary(img, faceCascade, 1.1, 10, color['blue'], "Training...")

    if len(coords) == 4:
        roi_img = img[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]
        generate_dataset(roi_img, u_id, name, img_id)
    
    return img


def main():

    name, u_id = data_entry()
    print("Webcam has started...")
    video_capture=cv2.VideoCapture(0, cv2.CAP_DSHOW)
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    if len(os.listdir(os.path.join("Data/", str(u_id) + ". " + name))) == 0:
        img_id = 0
    else:
        files_path = os.path.join("Data/", str(u_id) + ". " + name, "*")
        files = sorted(glob.iglob(files_path), key=os.path.getctime, reverse=True) 
        img_id = int((str(files[0]).rpartition(".")[0]).rpartition("\\")[2])+1

    while True:
        _, img = video_capture.read()
        img = detect (img, faceCascade, u_id, name, img_id)
        cv2.imshow("Face Detection", img)
        img_id += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if os.path.exists(os.path.join("Data/", str(u_id) + ". " + name, "Thumbs.db")):
        os.remove(os.path.join("Data/", u_id + ". " + name, "Thumbs.db"))


    video_capture.release()
    cv2.destroyAllWindows()
    classifier.train_classifier("data")

if __name__ == "__main__":
    main()