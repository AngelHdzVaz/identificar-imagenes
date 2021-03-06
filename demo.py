# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 09:57:55 2019

@author: USUARIO
"""

from __future__ import print_function, division
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision
from torchvision import datasets,transforms
import os

#transformacion de la imagen a tensor, escala de grises y el tamaño de 28x28

transform =transforms.Compose([
        torchvision.transforms.Grayscale(num_output_channels=1),
        torchvision.transforms.Resize((100,100), interpolation=2),
        transforms.ToTensor()
        ])

#ruta para identificar los datos de imagebes
data_dir = 'data/'

#{} esto identifica un diccionario
imageFolder = {}
for r in ['train','test']:
    x = os.path.join(data_dir, r)
    imageFolder[r] = (
            datasets.ImageFolder(root = x, transform = transform))
    
    
#cargar imagenes
    
dataLoaders = {}
for i in['train', 'test']:
    dataLoaders[i] = torch.utils.data.DataLoader(imageFolder[i], batch_size=3,shuffle=True,num_workers=0)
    
inputs, classes = next(iter(dataLoaders['train']))

out = torchvision.utils.make_grid(inputs)

#mostrar imagen y se transpone para que quede en la orientación correcta
def imshow(inp, title = None):
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)


#obtener el nombre de las clases
        
class_names = imageFolder['train'].classes
imshow(out, title=[class_names[x] for x in classes])

#y esto es lo mismo que el for anterior
# for x in class_names:
#   imshow(out,title=x)

import torch


