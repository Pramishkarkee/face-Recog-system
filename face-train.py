import os
from PIL import Image
import numpy as np 
import cv2
import pickle
current_id=0
x_train=[]
y_lable=[]
label_id={}
face_Cascade=cv2.CascadeClassifier('image/haarcascade_frontalface_default.xml')

recognizer=cv2.face.LBPHFaceRecognizer_create()
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
img_dir=os.path.join(BASE_DIR,"trainimage")
print(img_dir)
for root,dirs,files in os.walk(img_dir):
    for fil in files:
        if fil.endswith("png") or fil.endswith("jpg") or fil.endswith("jpeg"):
            path=os.path.join(root,fil)
            label=os.path.basename(os.path.dirname(path))
            if label in label_id:
                pass
            else:
                label_id[label]=current_id
                current_id +=1

            id_=label_id[label]
            print(label_id)
            pill_image=Image.open(path).convert("L") #grayscale
            # pill_image=cv2.resize(pill_image,(550,550))
            final_image=pill_image.resize((550,550),Image.ANTIALIAS)
            image_array=np.array(final_image,"uint8")
            # print(label,path)
            # print("image array",image_array) 
            face=face_Cascade.detectMultiScale(image_array,1.3,10)
            
            for (x,y,w,h) in face:
                roi=image_array[y:y+h,x:x+w]
                x_train.append(roi)
                y_lable.append(id_)

print("xtrain",x_train)
print("ylabel",y_lable)
with open("labels.pickle",'wb') as f:
    pickle.dump(label_id,f)
recognizer.train(x_train,np.array(y_lable))
recognizer.save("trainner.yml")