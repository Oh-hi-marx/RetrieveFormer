from groundingdino.util.inference import load_model, predict, annotate, Model
import cv2
from tqdm import tqdm 
import torch 
import torchvision
from PIL import Image
import groundingdino.datasets.transforms as T
import numpy as np 
import os 

class GroundingDino:
    def __init__(self,CONFIG_PATH = "./Groundingdino/GroundingDINO_SwinB.py",CHECKPOINT_PATH = "./Groundingdino/groundingdino_swinb_cogcoor.pth"):
        self.DEVICE = "cuda"
        self.transform =   T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ] )
        self.model = load_model(CONFIG_PATH, CHECKPOINT_PATH)

    def load_image(self,image):
        image_source = image.convert("RGB")
        image = np.asarray(image_source)
        image_transformed, _ = self.transform(image_source, None)
        return image, image_transformed

    def pred(self, image, TEXT_PROMPT, BOX_THRESHOLD, TEXT_THRESHOLD ):
        image_source, image = self.load_image(image)
        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            device=self.DEVICE,
        )
        return boxes, logits, phrases, image_source

    def annotate_frame(self,image_source, boxes, logits, phrases):
        boxes = self.xyxy2xywh(boxes)
        return annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

    def xyxy2xywh(self,x):
        # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
        y[:, 2] = x[:, 2] - x[:, 0]  # width
        y[:, 3] = x[:, 3] - x[:, 1]  # height
        return y


    def xywh2xyxy(self,x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y
        
    #Perform nms. Input boxes in xywh format. Returns xyxy format
    def nms(self,boxes, logits, phrases, iou_thresh = 0.5 ):
        boxes = self.xywh2xyxy(boxes) #boxes must be xyxy for nms
        #perform nms
        indexs  = torchvision.ops.nms(boxes, logits, iou_thresh)

        #select boxes, logits, phrases based on index
        boxes= torch.index_select(boxes, 0, indexs)
        logits= torch.index_select( logits, 0, indexs)
        phrases=  [phrases[i] for i in indexs]

        #boxes = self.xyxy2xywh(boxes) #back to xywh
        return boxes, logits, phrases

if __name__ == "__main__":
    #takes input folder, writes preds and boxes in yolo format
    files = os.listdir('images')
    os.makedirs("outputs", exist_ok=True)
    dino = GroundingDino(CONFIG_PATH = "./GroundingDINO_SwinB.py",CHECKPOINT_PATH = "./groundingdino_swinb_cogcoor.pth")
    for f in tqdm(files):
        image_path = 'images' + os.sep + f
        image = Image.open(image_path)

        
        TEXT_PROMPT = "camera, man, take, picture, mine, sea mine, catch, leather jacket, black, person, wear, photographer, lens, see, face, jacket, selfie, brown, room, use, cup, boy, eye, hand, close-up, church, video camera, hair, camera lens"
        TEXT_PROMPT = "leaf"
        BOX_THRESHOLD = 0.1
        TEXT_THRESHOLD =0.2

        #Predict boxes, confidence (logits) and phrases
        boxes, logits, phrases, image_source = dino.pred(image, TEXT_PROMPT, BOX_THRESHOLD,  TEXT_THRESHOLD)
        #perform non-max suppresions. This removes some phrases
        boxes, logits, phrases = dino.nms(boxes, logits, phrases)

        annotated_frame = dino.annotate_frame (image_source,boxes, logits, phrases)
        cv2.imwrite('outputs'+ os.sep +f, annotated_frame)
        with open('outputs' + os.sep+ f.rsplit(".",1)[0] +'.txt', 'w') as f:
            for box in list(boxes):
                box = (box.tolist())
                if(box[3]<0.5):
                    box=  str(box[0]) + " " + str(box[1]) +" " +str(box[2]) + " " +str(box[3]) 
                    f.write("0 " + " " + box+ "\n" )


