import cv2
import numpy as np
import face_recognition

imgelon= face_recognition.load_image_file('imagebasic/elon mask.jpg')
imgelon= cv2.cvtColor(imgelon,cv2.COLOR_BGR2RGB)
imgtest= face_recognition.load_image_file('imagebasic/elon mask test.jpg')

imgtest= cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)


faceloc=face_recognition.face_locations(imgelon)[0]
encodeelon=face_recognition.face_encodings(imgelon)[0]
cv2.rectangle(imgelon,(faceloc[3],faceloc[0],faceloc[1],faceloc[2]),(255,0,255),2)

facelocTest=face_recognition.face_locations(imgtest)[0]
encodeelonTest=face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest,(faceloc[3],faceloc[0],faceloc[1],faceloc[2]),(255,0,255),2)

res= face_recognition.compare_faces([encodeelon],encodeelonTest)
facedis= face_recognition.face_distance([encodeelon],encodeelonTest)
print(res,facedis)
cv2.putText(imgtest,f'{res} {round(facedis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('Elon mask',imgelon)
cv2.imshow('Elon mask',imgtest)
cv2.waitKey(0)

