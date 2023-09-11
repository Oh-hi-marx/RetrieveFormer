import numpy as np
import torch
import cv2
import os
from PIL import Image
from os import listdir
from os.path import isfile, join
from tqdm import tqdm 

#from samHQ.samHQ import SamHQ
from Groundingdino.grounding_dino import GroundingDino
from utils import cropMasks, xywhn2xywh

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
    
def xywhn2xywh(x, w=640, h=640):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * x[:, 0]# top left x
    y[:, 1] = h * x[:, 1]
    y[:, 2] = w * x[:, 2]
    y[:, 3] = h * x[:, 3] 
    return y

if __name__ == "__main__":
    mypath = 'inputs'
    output = 'outputs'

    TEXT_PROMPT = "anyobject, person, thing"
    BOX_THRESHOLD = 0.2
    TEXT_THRESHOLD = 0.1
    nms_iou_thresh = 0.5 

    minBox = 20 #minimum box size in pixels. ignore under this size 
    #load genertic object detector
    dino = GroundingDino()
    #load generic segmenation model
    #sam = SamHQ( sam_checkpoint = "./samHQ/sam_hq_vit_h.pth")

    #load all files
    onlyfiles = [os.path.join(mypath,f) for f in listdir(mypath) if isfile(join(mypath, f))]
    for i in tqdm(onlyfiles):
        try:
            image = Image.open(i).convert("RGB")
            #detect all objects
            boxes, logits, phrases, image_source = dino.pred(image, TEXT_PROMPT, BOX_THRESHOLD,  TEXT_THRESHOLD)
            boxes, logits, phrases = dino.nms(boxes, logits, phrases, nms_iou_thresh )
            #crop each object with sam
            w,h = image.size
            
            #boxesPixel = xywhn2xywh(boxes, w, h )  #covert from normalised to pixel values
            #boxesPixel = [list(np.array(f)) for f in boxesPixel]
            #boxesPixel.append(boxesPixel[0])

            #samMasks, samScores, samLogits = sam.pred(image, boxesPixel)
            #crop out mask, and also apply to rgb image
            #samMasks = cropMasks(samMasks, boxesPixel, np.array(image)[:, :, ::-1].copy(), minBox)
            samMasks= cropBoxes(image, boxes)
            fileoutpath = output + os.sep + i.rsplit(os.sep,1)[-1]
            os.makedirs(fileoutpath, exist_ok = True)
            print(fileoutpath)
            for j, mask in enumerate(samMasks):
               
                mask.save(fileoutpath+ os.sep  + str(j) + ".png")
            image.save(fileoutpath + os.sep + "original.png")
        except Exception as E:
            pass
    #print(masks)
    

