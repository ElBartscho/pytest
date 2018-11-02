import datetime
import numpy as np
import cv2
import sys

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('../Cascades/haarcascade_frontalface_default.xml')

if cap is None:
  sys.exit("no video capture available!")


while (True):
  # Capture frame-by-frame
  ret, frame = cap.read()
  faces = face_cascade.detectMultiScale(frame, 1.3)
  
  for (x,y,w,h) in faces:
    facepic = frame[y:y+h, x:x+w]
    
    if cv2.waitKey(1) & 0xFF == ord('s'):
      cv2.imwrite("pic.jpg", facepic)
    
    frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    
        
  cv2.imshow('Frame',frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
    
cv2.destroyAllWindows()
