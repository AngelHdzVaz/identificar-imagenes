# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 00:50:12 2019

@author: USUARIO
"""
import os
import tensorflow as tf
import torch
import numpy as np
import cv2
import glob
from torchvision import datasets
from pathlib import Path
import torchvision.transforms as transform
import re
import matplotlib.pyplot as plt
num_workers = 0
batch_size = 20
images=[]

#convertir imagenes a grises y 28*28
for filename in Path('.\imagenes').rglob('*.png'):
        img=cv2.imread(str(filename))
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)                         
        img=cv2.resize(img,(28,28))
        images.append(img)
        img=cv2.imwrite(str(filename),img)
#convertir a un vector cada imagen
numeros=[]
for i in images:
    a = i.reshape(1,784)
    numeros.append(a)
#empezamos leyendo las imagenes or carpeta    
dirname = os.path.join(os.getcwd(), 'imagenes')
imgpath = dirname + os.sep 
images = []
directories = []
dircount = []
prevRoot=''
cant=0
 
print("leyendo imagenes de ",imgpath)
 
for root, dirnames, filenames in os.walk(imgpath):
    for filename in filenames:
        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
            cant=cant+1
            filepath = os.path.join(root, filename)
            image = plt.imread(filepath)
            images.append(image)
            b = "Leyendo..." + str(cant)
            print (b, end="\r")
            if prevRoot !=root:
                print(root, cant)
                prevRoot=root
                directories.append(root)
                dircount.append(cant)
                cant=0
dircount.append(cant)
 
dircount = dircount[1:]
dircount[0]=dircount[0]+1
print('Directorios leidos:',len(directories))
print("Imagenes en cada directorio", dircount)
print('suma Total de imagenes en subdirs:',sum(dircount))

#creando etiquetas de cada imagen y a que clase pertenecen
labelstr=[]
indice=0
for cantidad in dircount:
    for i in range(cantidad):
        labelstr.append(indice)
    indice=indice+1
print("Cantidad etiquetas creadas: ",len(labelstr))


train_data = numeros
test_loader = numeros
#carga de datos de prueba y entrenamiento y etiquetas de imagenes, prueba de 20 imagene de entrenamiento y test
train_data = torch.utils.data.DataLoader(train_data,batch_size=batch_size)
label2 = torch.utils.data.DataLoader(labelstr,batch_size=batch_size)
train_loader = torch.utils.data.DataLoader(test_loader,batch_size=batch_size)

import matplotlib.pyplot as plt
dataiter = iter(train_loader)
images,label2 = dataiter.next()
images = images.numpy()

fig = plt.figure(figsize=(25,4))
for idx in np.arange(20):
    ax = fig.add_subplot(2,20/2,idx+1,xticks=[],yticks=[])
    ax.imshow(np.squeeze(images[idx]),cmaps='gray')
    ax.set_title(str(label2[idx].item()))