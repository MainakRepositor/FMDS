import cv2
from super_gradients.training import models
import torch
import numpy as np
import math

image = cv2.imread('sample/demo1.jpg')



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = models.get('yolo_nas_s' , pretrained_weights='coco').to(device)

classNames = ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
            'traffic light', 'fire hydrant', 'stop sign','parking meter', 'bench', 'bird',
            'cat', 'dog','horse', 'sheep','cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie','suitcase', 'frisbee', 'skis',
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
            'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
            'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
            'teddy bear', 'hair drier', 'toothbrush']






result = list(model.predict(image , conf=0.35))[0]
bbox_xyxys = result.prediction.bboxes_xyxy.tolist()
confidences = result.prediction.confidence
labels = result.prediction.labels.tolist()
for (bbox_xyxy , confidence , cls) in zip(bbox_xyxys , confidences , labels):
    bbox = np.array(bbox_xyxy)
    x1 , y1 , x2, y2 = bbox[0] , bbox[1], bbox[2], bbox[3]
    x1, y1, x2, y2 = int(x1) , int(y1), int(x2) , int(y2)
    classname = int(cls)
    classname = classNames[classname]
    conf = math.ceil(confidence*100)/100
    label = f'{classname}{conf}'
    
    t_size = cv2.getTextSize(label , 0 , fontScale=1, thickness=2)[0]
    
    c2 = x1 + t_size[0], y1 - t_size[1] -3
    
    cv2.rectangle(image , (x1,y1) , (x2, y2), [0,255,255] , 3)
    
    cv2.rectangle(image, (x1 , y1), c2, [255,144,30], -1 , cv2.LINE_AA)
    
    cv2.putText(image , label , (x1, y1-2), 0 , 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
    
resize_image = cv2.resize(image , (0,0), fx=0.4 , fy=0.4 , interpolation=cv2.INTER_AREA)



cv2.imshow('Frame', resize_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

