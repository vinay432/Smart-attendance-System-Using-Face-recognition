import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path='attendance'
images=[]
classNames=[]
mylist= os.listdir(path)
print(mylist)
for cl in mylist:
    currimg=cv2.imread(f'{path}/{cl}')
    images.append(currimg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodelist= []
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist


def markattendance(name):
    with open('attend.csv','r+') as f:
        myDatalist= f.readlines()
        namelist= []
        for line in myDatalist:
            entry=line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.now()
            dtstring=now.strftime("%m/%d/%Y, %H:%M:%S")
            f.writelines(f'\n{name},{dtstring}')

#makattendance('elon')

encodelistknown= findEncodings(images)
print("encoding completed")

cap= cv2.VideoCapture(0)
while True:
    success,img= cap.read()
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurrFrame = face_recognition.face_locations(imgS)
    encodeCurrFrame = face_recognition.face_encodings(imgS,facesCurrFrame)

    for encodeFace,faceloc in zip(encodeCurrFrame,facesCurrFrame):
        matches=face_recognition.compare_faces(encodelistknown,encodeFace)
        facedis=face_recognition.face_distance(encodelistknown,encodeFace)
        matchIndex= np.argmin(facedis)

        if matches[matchIndex]:
            name= classNames[matchIndex].upper()
            y1,x2,y2,x1= faceloc
            y1, x2, y2, x1= y1*4,x2*4,y2*4,x1*4

            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markattendance(name)

    cv2.imshow('webcam',img)
    cv2.waitKey(1)