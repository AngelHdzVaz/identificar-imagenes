# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 13:58:50 2019

@author: USUARIO
"""


import tensorflow as tf

# define a transform
import numpy as np
import cv2
import glob
from pathlib import path
import matplotlib.pyplot as plt
images=[]


for filename in path('imagenes').rglob('*.png'):
        img=cv2.imread(str(filename))
        img=cv2.cvtcolor(img,cv2.color_bgr2gray)                         
        img=cv2.resize(img,(28,28))
        images.append(img)
        img=cv2.imwrite(str(filename),img)

numeros=[]
for i in images:
    a = i.reshape(1,784)
    numeros.append(a)
