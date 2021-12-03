import cv2
import numpy as np
import tensorflow as tf
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = tf.keras.models.load_model('bestmodel.h5')
image_size = 48
Classes  = ['DISGUST','HAPPY','NEUTRAL','SAD','SURPRISE']
emotions = []


class Video(object):

    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__ (self):
        self.video.release()
    
    def get_frame(self):


        ret,test_img=self.video.read()# captures frame and returns boolean value and captured image 

        frame= cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)  

        gray_img= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
        
        faces_detected = faceDetect.detectMultiScale(gray_img,1.1,4)  
        
        
        for (x,y,w,h) in faces_detected:  
            cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7) 
                
                
                
            roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image  
            roi_gray=cv2.resize(roi_gray,(48,48))  
        #   print(roi_gray.shape)
            img = np.expand_dims(roi_gray,axis=0)
            img = img.reshape(1,48,48,1)
            pred = model.predict(img)
            emotion = Classes[np.argmax(pred[0])]
            emotions.append(emotion)
            cv2.putText(test_img, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            resized_img = cv2.resize(test_img, (1000, 700))  
            # cv2.imshow('Facial emotion analysis ',resized_img)
            key = cv2.waitKey(200)

        ret,jpg = cv2.imencode('.jpg',test_img)
        return jpg.tobytes()
    
    def close(self):
        self.video.release()  
        cv2.destroyAllWindows()
    
