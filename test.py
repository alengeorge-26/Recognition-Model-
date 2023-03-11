import cv2 as cv
import numpy as np

haar_cascade=cv.CascadeClassifier('/content/drive/MyDrive/OpenCV/haar_face.xml')

p=[]
DIR='/content/drive/MyDrive/OpenCV/Faces/train'
for i in os.listdir(DIR):
  p.append(i)

face_recognizer=cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img=cv.imread('/content/drive/MyDrive/OpenCV/Faces/val/madonna/1.jpg')

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv2_imshow(gray)

faces_rect=haar_cascade.detectMultiScale(gray,1.1,3)

for (x,y,w,h) in faces_rect:
        faces_roi=gray[y:y+h,x:x+w]

        label,confidence=face_recognizer.predict(faces_roi)
        print(f'{p[label]} with a confidence of {confidence}')
