import cv2 
import torch 
import os
from tqdm import tqdm 
import numpy as np 
from PIL import Image 

def xywhn2xywh(x, w=640, h=640):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * x[:, 0]# top left x
    y[:, 1] = h * x[:, 1]
    y[:, 2] = w * x[:, 2]
    y[:, 3] = h * x[:, 3] 
    return y

def padBackground(image, boxes,paddingSize = 2 ):
     #How much bigger background padding is. 
    w,h = image.size
    bgw, bgh = int(w*paddingSize), int(h*paddingSize)
    padStart = (paddingSize/paddingSize)/2 #Image is pasted in center of background

    #Pad with black background
    background = Image.new('RGB', (bgw, bgh))
    background.paste(image, (int(bgw*padStart), int(bgh*padStart)))

    #Add padding to boxes
    boxes /= paddingSize
    boxes += np.array([padStart,padStart, padStart, padStart]) 
    return boxes, background, bgw, bgh
    
#Crops expanded boxes from PIL image. Boxes are expected to be tensor or numpy array
def cropBoxes(image, boxes, expand = 0):
    #boxes are actually xyxy normalised. Convert to pixel values
    w,h = image.size
    boxes = xywhn2xywh(boxes, w, h)
    

    #Crop out boxes, with expansion if possible
    crops = []
    expandx = expand*w
    expandy = expand*h
    for box in boxes:     
        #Expand boxes as far as possible, without out of bounds
        box[0] = max(box[0] - expandx, 0)
        box[1] = max(box[1] - expandy, 0)
        box[2] = min(box[2] + expandx, w)
        box[3] = min(box[3] + expandy, h)
        box = np.round(np.array(box))
        crop = image.crop(box)
        crops.append(crop)
    return crops


def resizeSide( img, maxSize= 512, force =False):
    #resize by longest side. Keep ratio 
    w, h = img.size
    ratio =  max(w,h)/ maxSize
    if(ratio>1 or force ): #only resize if image larger than max size 
        newSize = [int(w/ratio), int(h/ratio)]
        img = img.resize(newSize)
    return img 

def cropMasks(masks, boxes, image, minBox):
    crops =[]
    for i, box in enumerate(boxes):
        box = np.round(np.array(box))
        boxSize = max((box[3]-box[1]), (box[2]-box[0]))
        #skip tiny tiny objects
        if(boxSize > minBox):
            crop = masks[i][int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            imagecrop = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            #add alpha 
            imagecrop =cv2.cvtColor(imagecrop , cv2.COLOR_RGB2RGBA) 
            imagecrop[:,:,3] = crop.astype(np.uint8)*255 
            crops.append(imagecrop)
    return crops

if __name__ == "__main__":

    pass