#Register in DB
import cv2
import os
import numpy as np,sys
from PIL import Image
import posix
class faceRecogEigen:
    face_cascade=cv2.CascadeClassifier('/home/sneha/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
#eye_cascade=cv2.CascadeClassifier('/home/shivani/opencv/data/haarcascades/haarcascade_eye.xml')
    cap=cv2.VideoCapture(0)
    print cap.isOpened()
    if not(cap.isOpened()): 
        cap.open() 
    
    id=input('Enter the user id : ')
    sampleNum=0;
    while True:
        ret,img=cap.read()
        #img=img[200:400,100:300]
        #gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        minisize = (img.shape[1],img.shape[0])
        miniframe = cv2.resize(img, minisize)
        gray=cv2.cvtColor(miniframe,cv2.COLOR_BGR2GRAY) #Converting the image into grayscale
        
        
        faces=face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
        for(x,y,w,h) in faces :
            sampleNum=sampleNum+1;
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)  #Blue color 255,0,0
            f=cv2.resize(gray[y:y+h,x:x+w],(200,200))
           # f=cv2.resize(gray[y:y+h,x:x+w],(200,200))
            equ=cv2.equalizeHist(f) #Histogram equalization
            blur = cv2.bilateralFilter(equ,0,20.0,22.0) #Smoothing
            

        cv2.imwrite("/home/sneha/faces/User."+str(id)+"."+str(sampleNum)+".pgm",blur) #Saving the captured images as User.id.samplenumber
        cv2.imshow('img',img)
	#k=cv2.waitKey(0)    #infinte waiting
        cv2.waitKey(100) #if i press q it'll break
        if(sampleNum>25):
            break	

    cap.release()
    cv2.destroyAllWindows()	
