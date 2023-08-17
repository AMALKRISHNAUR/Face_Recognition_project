from model import create_model
nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('weights/nn4.small2.v1.h5')
empeding = []
import os
from PIL import Image
import numpy as np
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    for imagePath in imagePaths:
        print(imagePath)
        PIL_img = Image.open(imagePath).convert('RGB')
        PIL_img = PIL_img.resize((96,96))
        img_numpy = np.array(PIL_img)
        img_numpy = (img_numpy / 255.).astype(np.float32)
        empeding.append(nn4_small2_pretrained.predict(np.expand_dims(img_numpy, axis=0))[0])
        # print(img_numpy)
        print(imagePath)
    return empeding
database = getImagesAndLabels('./database')
#database={"seetal":database[0],"sreenath":database[1],"Amal":database[2],"jayasree":database[3]}
database={"Amal":database[0]}
def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))
import cv2
faceCascade = cv2.CascadeClassifier('cascade/haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX
cam = cv2.VideoCapture(0)
cam.set(3,1080)
cam.set(3,720)

minW = 0.1*cam.get(3)

minH = 0.1*cam.get(4)

while True:
    ret,img = cam.read()
    new_img = img
    dis = []

    faces = faceCascade.detectMultiScale(img,1.2,5,minSize=(int(minW),int(minH)))
    for (x,y,w,h) in faces:
        print(f"{x},{ y},{w},{h}")
        if img is None:
            print('Wrong path:', path)
            continue
        else:
            try:
                img=img[y-70:y+h+30,x-10:x+w+10]
                img = cv2.resize(img,(96,96),interpolation=cv2.INTER_CUBIC)
                img = (img / 255.).astype(np.float32)
                encod = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]
            except:
                continue
        min_dist = 100
    
    #Loop over the database dictionary's names and encodings.
        for (name, db_enc) in database.items():
        
        # Compute L2 distance between the target "encoding" and the current db_enc from the database. (≈ 1 line)
            dist = np.linalg.norm(encod - db_enc)

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
            if dist < min_dist:
                min_dist = dist
                identity = name
    
    # YOUR CODE ENDS HERE
    
        if min_dist > 0.6:
            print("Not in the database.")
            cv2.putText(
                new_img,
                "Note Recognized",
                (x+5,y-5),
                font,
                1,
                (255,255,255),
                2)
        else:
            cv2.putText(
                new_img,
                str(identity),
                (x+5,y-5),
                font,
                1,
                (255,255,255),
                2)
            print ("it's " + str(identity) + ", the distance is " + str(min_dist))
            
        
            
    cv2.imshow("camera",new_img)
    k = cv2.waitKey(10) &0xff
    if k == 27:
        break
print("All Done")
cam.release()
cv2.destroyAllWindows()
