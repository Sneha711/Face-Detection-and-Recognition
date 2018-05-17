import posixpath
import os #Importing os for file path
import cv2
import numpy as np
from PIL import Image #Importing python image library

path="/home/sneha/faces"  #Path to the folder in which the training faces are stored

#Method to get images and ids, this function is for preparing the training data
def getImagesWithID(path):
    imagePaths=[posixpath.join(path,f) for f in os.listdir(path)] #Get all file path
    facesh=[] #Initializing empty face list
    IDs=[] #Initializing empty face list
    for imagePath in imagePaths:
        #print(imagePath)
        faceImg=Image.open(imagePath).convert('L') #Converting the images into grayscale 
        faceNp=np.array(faceImg,'uint8') #PIL image to numpy array
        
        ID=int(os.path.split(imagePath)[-1].split('.')[1]) #Splitting the path and filename to get the id
        
        facesh.append(faceNp) #add faces to facesh list
        #print(ID)
        IDs.append(ID) #add ids to IDs list
        #cv2.imshow("training",faceNp)
        #cv2.waitKey(10)
    return IDs,facesh 


def facerec():
    Ids,facesh=getImagesWithID(path) 
    recognizer = cv2.face.createEigenFaceRecognizer() #creating EigenFaceRecognizer
    recognizer.train(facesh,np.array(Ids)) #Training the model using faces and ids(training data)
    
    cascadePath = "/home/sneha/opencv/data/haarcascades/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath); # Creating classifier from prebuilt model
    cam = cv2.VideoCapture(0)
    #font = cv2.InitFont(cv2.FONT_HERSHEY_SIMPLEX, 1, 1, 0, 1, 1)
    while True:
        ret,im =cam.read()
        #print(im.shape[0],im.shape[1])
        minisize = (im.shape[1],im.shape[0])
        miniframe = cv2.resize(im, minisize)
        #print(im.shape[0],im.shape[1])
        gray=cv2.cvtColor(miniframe,cv2.COLOR_BGR2GRAY)
        #gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.3,5)
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)  # Creating rectangle around the face
            #f=cv2.resize(gray[y:y+h,x:x+w],(200,200))
            f=cv2.resize(gray[y:y+h,x:x+w],(200,200))
            Id, conf = recognizer.predict(f) # Recognize which face belongs to which ID
            #print(h)
            cv2.putText(im,str(Id), (x+5,y+h-20),cv2.FONT_HERSHEY_SIMPLEX,2, 255,3)  # Displays the id in the rectangle
        cv2.imshow('face',im)  # Display the video frame with the bounded rectangle
        if cv2.waitKey(10) & 0xFF==ord('q'): #Press q to exit
            break
    cam.release()
    cv2.destroyAllWindows()

#Calling the facerec() function
facerec()
