import cv2 as cv
from keras.models import load_model
from time import sleep
from tensorflow.keras.utils import img_to_array
from keras.preprocessing import image
import numpy as np

face_classifier=cv.CascadeClassifier(r"C:\Users\HP\Desktop\Emotion_Detection_using_CNN\harcascade_frontalface.xml")
classifier=load_model(r"C:\Users\HP\Desktop\Emotion_Detection_using_CNN\model.h5")

emotion_labels=['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

cap=cv.VideoCapture(0)

while True:
    _,frame=cap.read()
    labels=[]
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray)

    for x,y,h,w in faces:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_gray=cv.resize(roi_gray,(48,48),interpolation=cv.INTER_AREA)


        if(np.sum([roi_gray])!=0):
            roi=roi_gray.astype('float')/255.0
            #converting it into array as our model is trained on array 
            roi=img_to_array(roi) 
            roi=np.expand_dims(roi,axis=0)

            predict=classifier.predict(roi)[0]
            label=emotion_labels[predict.argmax()]
            label_pos=(x,y-11)
            cv.putText(frame,label,label_pos,cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

        else:
            cv.putText(frame,"No Faces",label_pos,cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    cv.imshow("Emotion Detector",frame)
    a=cv.waitKey(50)
    if(a=="q"):
        break

cap.release()
cv.destroyAllWindows()
    
