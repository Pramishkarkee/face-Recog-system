import cv2
import numpy as np 
import pickle
cap=cv2.VideoCapture(0)
face_Cascade=cv2.CascadeClassifier('image/haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('imageRec/haarcascade_eye_tree_eyeglasses.xml')
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner.yml')
labels={"person_name":1}
with open("labels.pickle",'rb') as f:
    og_label=pickle.load(f)
    labels={v:k for k,v in og_label.items()}
while True:
    success,img=cap.read()
    # print(success)
    gray =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face=face_Cascade.detectMultiScale(gray,1.3,10)
    # m_f=cv2.imread("image/my_face.png")
    # cv2.imshow("myface",m_f)
    for (x,y,w,h) in face:
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=gray[y:y+h,x:x+w]
        #machin learning
        id_,conf=recognizer.predict(roi_gray)
        if conf>=45 :
            print("id",id_ ,labels[id_])
            cv2.putText(img,labels[id_],(x,y),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,255),2)
        cv2.imwrite("my_face.png",roi_gray)
        # cv2.imshow("myface",my_face.png)
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
        eyes=eye_cascade.detectMultiScale(roi_gray)
        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(255,0,0),3)
    
    cv2.imshow("capture image",img)
    
    if cv2.waitKey(1)==ord('q'):
        break

cv2.destroyAllWindows()