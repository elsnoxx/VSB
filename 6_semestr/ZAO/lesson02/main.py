import cv2 as cv
import numpy as np
import os

path = "test-images"

if(os.path.exists("out-red") == False):
    os.mkdir("out-red")
        
if(os.path.exists("out-green") == False):
    os.mkdir("out-green")

files = []
for file in os.listdir(path):
    print(file)
    files.append(os.path.join(path, file))
    
result = {}
#### 1 
for file in files:
    print(file)
    image = cv.imread(file)
    image = cv.resize(image, (300, 600))

    b, g, r = cv.split(image)

    _, threshRed = cv.threshold(r, 100, 255, cv.THRESH_BINARY)
    redPixels = cv.countNonZero(threshRed)
    _, threshGreen = cv.threshold(g, 100, 255, cv.THRESH_BINARY)
    greenPixels = cv.countNonZero(threshGreen)

    print(redPixels)
    print(greenPixels)

    if(greenPixels > redPixels):
        result[file] = "red"
    else:
        result[file] = "green"

print(result)
# Display each channel
# cv.imshow("Blue Channel", b)
# cv.imshow("Green Channel", g)
# cv.imshow("Red Channel", r)
# cv.imshow("image", image)
# cv.waitKey()


