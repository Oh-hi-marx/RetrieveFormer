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
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, ConvNextV2ForImageClassification
import random
import matplotlib.pyplot as plt

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

def loadData(path):
    folders = [path + os.sep + f for f in os.listdir(path)]
    allFolders =  []
    for folder in folders:
        files = [ folder + os.sep + f for f in os.listdir(folder)]
        allFolders.append({'root':folder, 'files':files})
    return allFolders

def PILtransform(img):
    if(random.random()>0.5):
        img = ImageOps.flip(img)
    if(random.random()>0.5):
        img = ImageOps.mirror(img)
    if(random.random()>0.5):
        img = img.rotate(random.randrange(0,180))
    if(random.random()>0.5):
        enhancer = ImageEnhance.Sharpness(img)
        enhancer.enhance(random.randrange(0,2))
    if(random.random()>0.5):
        enhancer = ImageEnhance.Contrast(img)
        enhancer.enhance(random.randrange(0,2))
    if(random.random()>0.5):
        img = img.filter(ImageFilter.BLUR)
    if(random.random()>0.5):
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.randrange(5,20)/10)

    return img

def resize(img, maxSide ):
    w,h = img.size
    longest = max(w,h)
    ratio = longest / maxSide
    return img.resize((int(w/ratio), int(h/ratio)))

class dataset(Dataset):
    def __init__(self, allFolders,transforms,vizDataloader= False):
        self.folders = allFolders
        self.vizDataloader = vizDataloader
        self.transforms = transforms
        self.maxPaste = 0.05 #highest xy coordinates of random paste. Too high, and object will be cut off
        self.objectSize = 200
        self.backgroundSize = 640
    def __len__(self):
        return (len(self.folders))

    def __getitem__(self, i):
        try:
            files = self.folders[i]
            root = files['root']
            files = files['files']

            #background image
            backgroundPath = root + os.sep + "original.png"
            img = Image.open(backgroundPath)
            img = PILtransform(img)
            #resize and pad to square
            img = resizePaste(img, self.backgroundSize,1)


            w,h = img.size

            #random paste location
            label = 0
            if(random.random()> 0.5 and len(files)):
            #get a random object from original image (Retreival = 1)
                randObject = Image.open( files[random.randrange(0, len(files))]).convert("RGBA")
                randObject = PILtransform(randObject)
                randObject = resizePaste(randObject, self.objectSize)
                img.paste(randObject,(0,0))
                label = 1
            else:
            #Should not contain a retrival object from original
                #if(random.random()> 0.5):
                #contains object from another random image
                randIdx = random.randrange(0, len(self.folders))
                if(randIdx!=i): #check not same idx as original
                    files = self.folders[randIdx]
                    files = files['files']
                    if(len(files)):
                        randObject = Image.open( files[random.randrange(0, len(files))]).convert("RGBA")
                        randObject = resize(randObject, self.objectSize)
                        randObject = PILtransform(randObject)
                        square= Image.new('RGB', (self.objectSize, self.objectSize))
                        img.paste(square, (0,0))
                        img.paste(randObject,(0,0))


            img = img.convert("RGB")
            if(self.vizDataloader):
                os.makedirs('sampleDataloader', exist_ok=True)
                img.save("sampleDataloader" + os.sep + str(random.randrange(0,10000))+ "_"+str(label) + '.jpg')
        except:
            img = Image.new('RGB', (self.objectSize, self.objectSize))
            label =0

        img = self.transforms(img)

        if(label):
            label = torch.FloatTensor([1, 0])
        else:
            label = torch.FloatTensor([0,1])
        return img, label


epochs  = 1000
numOutput=2
trainDir = 'outputs'
valDir = 'outputs'
os.makedirs('weights', exist_ok = True)
VIZDATALOADER = 0
learn_rate = 0.00001
modelPath = None#'weights/61.pth'
#load model
net = timm.create_model('timm/eva02_base_patch14_448.mim_in22k_ft_in22k', pretrained=True)


data_config = timm.data.resolve_model_data_config(net)

transformTrain = timm.data.create_transform(**data_config, is_training=True)
transform = timm.data.create_transform(**data_config, is_training=False)

net.head  = nn.Sequential(nn.Linear(net.head.in_features, numOutput), 
                                 nn.Sigmoid())
    
batch_size = 8

allFoldersTrain = loadData(trainDir)
trainset = dataset(allFoldersTrain ,transformTrain, vizDataloader= VIZDATALOADER)
allFoldersVal = loadData(valDir)
testset = dataset(allFoldersVal ,transform)
print(len(allFoldersTrain))

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)




device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if(modelPath):
                             
    net.load_state_dict(torch.load(modelPath))
    print("loading", modelPath)

net.to(device)
net.train()
import torch.optim as optim


criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=learn_rate, momentum=0.9)

losses = []
for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if(i%1000 ==0):
            print(loss.item())

    running_loss/= batch_size
    print("epoch: ", epoch, " loss: ", running_loss)
        #running_loss = 0.0
    losses.append(running_loss)

    PATH = "weights" + os.sep + str(epoch) + '.pth'
    torch.save(net.state_dict(), PATH)

    ### Plot loss ###
    xs = [x for x in range(len(losses))]
    plt.plot(xs, losses)
    plt.savefig("loss.jpg")

print('Finished Training')





correct = 0
total = 0

# since we're not training, we don't need to calculate the gradients for our outputs
accuracy=0
counter = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)

        # calculate outputs by running images through the network
        outputs = net(images)

        error =torch.abs(labels- outputs)<0.5
        accuracy += (torch.count_nonzero(error) / error.size(dim=0) ).cpu().numpy()
        counter+=1

print("Test accuracy", accuracy/counter)

