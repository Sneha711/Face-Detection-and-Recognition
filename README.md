# Face-Detection-and-Recognition
Code for Face Detection and Recognition using Eigenfaces Algorithm with OpenCV in python.

### Prerequisites

* Python 2.7
* OpenCV
* Other requirements are listed in requirements.txt file and can be installed by running the following command: 
`pip install -r requirements.txt`


### How it Works

* Step 1
`trainerEigen.py`: Prepares training data. 
1] Detects faces in the video frame.
2] Assign each detected face an integer label(id) of the person it belongs to.
3] Crops the detected faces in the frame and saves it in a folder as User.id.samplenumber
4] 25 images for each person is taken.

* Step 2
`faceRecogEigen.py`:  
1] Trains OpenCV's EigenFace Recognizer by feeding the data prepared in Step 1.
2] Performs prediction on Real time video frame.






