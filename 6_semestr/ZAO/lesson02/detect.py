# TODO pip install ultralytics

import cv2 as cv
from ultralytics import YOLO
from check_utils import desizionModel, calculateAccurency, loadFiles
import numpy as np


# Load the YOLOv11 model
model = YOLO('tlight-v11.pt')

cv.namedWindow('frame', 0)

path = "02-cv-template-v2-yolo/test-big-zao"

files = loadFiles(path)

for file in files:
    frame = cv.imread(file)
    frame_paint = frame.copy()

    # Run YOLOv11 inference on the frame
    results = model.predict(frame, imgsz=480, conf=0.2)
    for box in results[0].boxes:
        print(box)
        x1, y1, x2, y2 = box.xyxy[0]
        conf = box.conf[0]

        cropt_img = frame[int(y1):int(y2), int(x1):int(x2)]
        desicion = desizionModel(cropt_img)
        
        if desicion == "red":
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        cv.putText(frame, f"Conf: {conf:.2f}", (int(x1), int(y1) - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
            
    cv.imshow('frame', frame)        
    cv.waitKey()      
