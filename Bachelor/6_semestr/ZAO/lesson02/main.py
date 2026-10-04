import os
import cv2 as cv
from check_utils import desizionModel, calculateAccurency, loadFiles

path = "test-images"
path_output_red = "out-red"
path_output_green = "out-green"

if(os.path.exists(path_output_red) == False):
    os.mkdir(path_output_red)
        
if(os.path.exists(path_output_green) == False):
    os.mkdir(path_output_green)

files = loadFiles(path)
    
result = {}
for file in files:
    image = cv.imread(file)
    desicion = desizionModel(image)
    result[file] = desicion
    
    if (desicion == "red"):
        cv.imwrite(os.path.join(path_output_red ,f"{os.path.basename(file)}.jpg"), image)
    else:
        cv.imwrite(os.path.join(path_output_green ,f"{os.path.basename(file)}.jpg"), image)

calculateAccurency(result)



