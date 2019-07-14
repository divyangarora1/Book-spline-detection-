# -*- coding: utf-8 -*-
"""
Created on Thu May 23 03:29:24 2019

@author: Parth Bhandari
"""

import cv2
import numpy as np
from sklearn.externals import joblib
from keras.preprocessing import image

dic = {1 : 'a', 2 : 'b', 3 : 'c', 4 : 'd',
       5 : 'e', 6 : 'f', 7 : 'g', 8 : 'h',
       9 : 'i', 10 : 'j', 11 : 'k', 12 : 'l',
       13 : 'm', 14 : 'n', 15 : 'o', 16 : 'p',
       17 : 'q', 18 : 'r', 19 : 's', 20 : 't',
       21 : 'u', 22 : 'v', 23 : 'w', 24 : 'x',
       25 : 'y', 26 : 'z', 27 : '0', 28 : '1',
       29 : '2', 30 : '3', 31 : '4', 32 : '5',
       33 : '6', 34 : '7', 35 : '8', 36 : '9',
       37 : 'A', 38 : 'B', 39 : 'C', 40 : 'D',
       41 : 'E', 42 : 'F', 43 : 'G', 44 : 'H',
       45 : 'I', 46 : 'J', 47 : 'K', 48 : 'L',
       49 : 'M', 50 : 'N', 51 : 'O', 52 : 'P',
       53 : 'Q', 54 : 'R', 55 : 'S', 56 : 'T',
       57 : 'U', 58 : 'V', 59 : 'W', 60 : 'X',
       61 : 'Y', 62 : 'Z'}


joblib_file = "minor2.pkl"
joblib_model = joblib.load(joblib_file)
Image=cv2.imread("abc.jpg")

G_Image=cv2.cvtColor(Image,cv2.COLOR_RGB2GRAY)
#Otsu Thresholding
blur = cv2.GaussianBlur(G_Image,(1,1),0)
ret,th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
image1,contours,hierarchy = cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
counter=0
dim = (64, 64)

for contour in contours:
    [x, y, w, h] = cv2.boundingRect(contour)
    if w/h > 2 or h>25 or h<5:
        continue

    try:
        resized = cv2.resize(Image[y-8:y + h+8, x-4:x + w+4], dim, interpolation=cv2.INTER_AREA)
        z = image.img_to_array(resized)
        z = np.expand_dims(z, axis=0)
        classes = joblib_model.predict_classes(z)
        if classes[0]==0:

            resized = cv2.resize(Image[y - 8:y + h + 8, x - 4:x + w + 4], dim, interpolation=cv2.INTER_CUBIC)

            z = image.img_to_array(resized)
            z = np.expand_dims(z, axis=0)
            classes = joblib_model.predict_classes(z)
            print(dic.get(classes[0] + 1))
            cv2.imwrite("save/" + str(counter) + '.jpg', resized)
        else:
            print(dic.get(classes[0]+1))
            cv2.imwrite("save/"+str(counter)+'.jpg', resized)

        if classes[0]>0:
            cv2.putText(Image, dic.get(classes[0]+1), (x-2, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255, thickness=2)
        counter+=1

    except Exception as e:
        print(str(e))


cv2.imshow('image',Image)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()