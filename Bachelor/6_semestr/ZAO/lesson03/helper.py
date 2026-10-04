import pyautogui
import cv2 as cv
import os
import time

def click_on_target(x, y):
    pyautogui.click(x, y, _pause=False)
    print(f"Klikám... {x} a {y}")
    
def load_files(path):
    red_files = []
    blask_files = []
    for file in os.listdir(path):
        if(file.split('_')[0] == 'red' ):
            red_files.append(os.path.join(path, file))
        elif(file.split('_')[0] == 'black' ):
            blask_files.append(os.path.join(path, file))  
    return red_files, blask_files  

def load_picture(name, scale_factor):
    img = cv.imread(name, 0)
    img = cv.resize(img, (int(img.shape[1] // scale_factor), int(img.shape[0] // scale_factor)))
    return img

def center_point(img):
    h, w = img.shape[:2]
    w = w // 2
    h = h // 2
    return h, w

def monitor_size():
    mon_width, mon_height = pyautogui.size()
    mon_width = mon_width // 2
    return mon_width, mon_height

def save_pic(path, image):
    if(os.path.exists(path) == False):
        os.mkdir(path)
    result = cv.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
    filename = time.strftime("%Y%m%d_%H%M%S") + ".jpg"
    cv.imwrite(os.path.join(path ,filename), result)
    
def find_template(image, template, threshold=None):
    res = cv.matchTemplate(image, template, cv.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv.minMaxLoc(res)
    if threshold is None:
        return max_val, max_loc
    return (max_val, max_loc) if max_val >= threshold else (max_val, None)

def get_roi(image, location, template_shape, margin=100):
    h, w = template_shape[:2]
    top = max(0, location[1] - margin)
    bottom = min(image.shape[0], location[1] + h + margin)
    left = max(0, location[0] - margin)
    right = min(image.shape[1], location[0] + w + margin)
    
    # Vrátíme výřez a souřadnice posunu, abychom to pak mohli přepočítat zpět
    return image[top:bottom, left:right], (left, top)