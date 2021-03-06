import numpy as np
import cv2
import sys

face_cascade = cv2.CascadeClassifier('../Cascades/haarcascade_frontalface_default.xml')
# Set Datapath to a Picture
img = cv2.imread('../../Picture.jpg')

if img is None:
  sys.exit("no image avalable!")

faces = face_cascade.detectMultiScale(img, 1.3)
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_color = img[y:y+h, x:x+w]

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

