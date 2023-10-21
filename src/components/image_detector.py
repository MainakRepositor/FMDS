import cv2
from ultralytics import YOLO
import math

def load_yolonas_process_each_image(image , confidence , st):
    
    model = YOLO('models/best_3.pt')
    classNames = ['proper_mask','no_mask','improper_mask']
    detections = model.predict(image , conf=confidence)[0]
    
    for detection in detections.boxes.data.tolist():
        
        x1, y1, x2, y2, conf, class_id = detection
        x1, y1, x2, y2 = int(x1) , int(y1), int(x2) , int(y2)
        classname = classNames[int(class_id)]
    
        conf = math.ceil(conf*100)/100
        label = f'{classname}{conf}'
        
        t_size = cv2.getTextSize(label , 0 , fontScale=1, thickness=2)[0]
        
        c2 = x1 + t_size[0], y1 - t_size[1] -3
        
        cv2.rectangle(image , (x1,y1) , (x2, y2), [0,255,255] , 3)
        
        cv2.rectangle(image, (x1 , y1), c2, [255,144,30], -1 , cv2.LINE_AA)
        
        cv2.putText(image , label , (x1, y1-2), 0 , 1, [255, 255, 255], thickness=2, lineType=cv2.LINE_AA)
        
    # resize_image = cv2.resize(image , (0,0), fx=0.4 , fy=0.4 , interpolation=cv2.INTER_AREA)

    st.subheader('Output Image')
    st.image(image, channels ='BGR', use_column_width = True)

    # cv2.imshow('Frame', resize_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()