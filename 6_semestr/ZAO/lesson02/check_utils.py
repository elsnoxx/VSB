import cv2 as cv
import numpy as np
import os

def loadFiles(path):
    files = []
    for file in os.listdir(path):
        files.append(os.path.join(path, file))
    return files

def calculateAccurency(result):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for res in result:
        correct = os.path.basename(res).split('-')[0] 

        if correct == "red" and result[res] == "red":
            tp += 1
        elif correct == "red" and result[res] == "green":
            fn += 1
        elif correct == "green" and result[res] == "red":
            fp += 1
        elif correct == "green" and result[res] == "green":
            tn += 1

    print("TP:", tp)
    print("TN:", tn)
    print("FP:", fp)
    print("FN:", fn)

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    print("Accuracy:", accuracy)

def desizionModel(img):
    isRed = 0
    isGreen = 0
    image = cv.resize(img, (300, 600))
    
    b, g, r = cv.split(image)
        
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    
    lower_red1 = np.array([0, 150, 150])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([160, 150, 150])
    upper_red2 = np.array([179, 255, 255])

    mask_red1 = cv.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv.inRange(hsv, lower_red2, upper_red2)
    
    lower_green = np.array([35, 150, 150])
    upper_green = np.array([85, 255, 255])

    kernel_dilate = np.ones((7,7), np.uint8)
    kernel_erode = np.ones((11,11), np.uint8)

    mask_red = mask_red1 + mask_red2
    mask_green = cv.inRange(hsv, lower_green, upper_green)

    mask_red = cv.dilate(mask_red, kernel_dilate, iterations=1)
    mask_red = cv.erode(mask_red, kernel_erode, iterations=1)

    mask_green = cv.dilate(mask_green, kernel_dilate, iterations=1)
    mask_green = cv.erode(mask_green, kernel_erode, iterations=1)

    
    count_red = cv.countNonZero(mask_red)    
    count_green = cv.countNonZero(mask_green)
    
    
    ratio_red = count_red / (count_red + count_green + 1)
    if ratio_red > 0.8:
        return "red"
    else:
        return "green"

        
    