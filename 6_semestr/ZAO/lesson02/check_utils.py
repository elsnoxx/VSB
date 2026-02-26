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

    _, threshRed = cv.threshold(r, 150, 255, cv.THRESH_BINARY)
    redPixels = cv.countNonZero(threshRed)
    _, threshGreen = cv.threshold(g, 150, 255, cv.THRESH_BINARY)
    greenPixels = cv.countNonZero(threshGreen)
    
    if(greenPixels > redPixels):
        isGreen += 1
    else:
        isRed += 1
        
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    
    lower_red1 = np.array([0, 150, 150])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([160, 150, 150])
    upper_red2 = np.array([179, 255, 255])

    mask_red1 = cv.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv.inRange(hsv, lower_red2, upper_red2)

    mask_red = mask_red1 + mask_red2
    count_red = cv.countNonZero(mask_red)
    
    lower_green = np.array([35, 150, 150])
    upper_green = np.array([85, 255, 255])

    mask_green = cv.inRange(hsv, lower_green, upper_green)
    count_green = cv.countNonZero(mask_green)
    
    
    ratio_red = count_red / (count_red + count_green + 1)
    if ratio_red > 0.6:
        isRed += 2
    else:
        isGreen += 2
        

    if(isGreen > isRed):
        return "green"
    else:
        return "red"
    