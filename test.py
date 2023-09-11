from random import shuffle
from tkinter.filedialog import LoadFileDialog
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os 
import timm
from torch.utils.data import DataLoader, Dataset
from os import path
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, ConvNextV2ForImageClassification
import random
from PIL import ImageDraw

def resize(img, maxSide ):
    w,h = img.size
    longest = max(w,h)
    ratio = longest / maxSide
    return img.resize((int(w/ratio), int(h/ratio)))

def resizePaste(img, size, right = 0):
    img = resize(img, size)
    square= Image.new('RGB', (size, size))
    if(right):
        w,h = img.size

        offsetw = size-w
        offseth = size -h
        square.paste(img, (offsetw,offseth))
    else:
        square.paste(img, (0,0))
    return square


modelPath = 'weights/623.pth'
path = 'testimages'

numOutput=2
trainDir = 'outputs'
valDir = 'outputs'
os.makedirs('weights', exist_ok = True)
batch_size = 32
backgroundSize = 640 
targetSize = 200


files= [path + os.sep + f for f in os.listdir(path) if os.path.isfile(path + os.sep + f )]
targets =  [path + os.sep +'targets' + os.sep + f for f in os.listdir(path + os.sep + "targets") if os.path.isfile(path + os.sep +'targets' + os.sep + f)]

#load model
net = timm.create_model('timm/eva02_base_patch14_448.mim_in22k_ft_in22k', pretrained=True)



data_config = timm.data.resolve_model_data_config(net)

transformTrain = timm.data.create_transform(**data_config, is_training=True)
transform = timm.data.create_transform(**data_config, is_training=False)

net.head  = nn.Sequential(nn.Linear(net.head.in_features, numOutput), 
                                 nn.Sigmoid())
                                 
net.load_state_dict(torch.load(modelPath))



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)
net.eval()



correct = 0
total = 0

os.makedirs(path  + os.sep + "preds", exist_ok=True)
# since we're not training, we don't need to calculate the gradients for our outputs
accuracy=0
counter = 0
targetImages = [resizePaste(Image.open(f), targetSize) for f in targets]
with torch.no_grad():
    for file in files:
        imageOG = Image.open(file)
        imageOG = resizePaste(imageOG, backgroundSize, 1)

        for target in targetImages:
        #paste onto square

            imageOG.paste(target)
            image = transform(imageOG).unsqueeze(0)
            images= image.to(device)

            # calculate outputs by running images through the network
            outputs = net(images)
            output = outputs[0].cpu().numpy()
            I1 = ImageDraw.Draw(imageOG)
            
            # Add Text to an image
            I1.text((0, 0), str(output), fill=(255, 0, 0))
            output = round(output[0]/ sum(output),3)
            imageOG.save(path+os.sep +"preds" + os.sep + file.split(os.sep)[-1].rsplit(".",1)[0] +"_"+ str(output) +'.jpg')



