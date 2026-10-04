from mss import mss
import cv2 as cv
import numpy as np
import pyautogui
import time
import os
from helper import click_on_target, load_files, load_picture, center_point, monitor_size, save_pic, find_template, get_roi

#big screen
scale_factor = 1.5
# scale_factor = 6

# scale_factor_other = 1
scale_factor_other = 1

show_img_result = False
fail_count = 0

path = 'duckhunt/tragets'  
path_debug = "debug"
cnt = 0 
cnt_rety = 5


restrat = load_picture('duckhunt/restart.png', scale_factor_other)
restrat_h, restrat_w = center_point(restrat)

 
red_files, blask_files = load_files(path)
templates = []
for file in blask_files:
    template = cv.imread(file, 0)
    template = cv.resize(template, (int(template.shape[1] * scale_factor), int(template.shape[0] * scale_factor)))
    templates.append(template)

mon_width, mon_height = monitor_size()

search_area = {"top": 0, "left": mon_width, "width": mon_width, "height": mon_height}

template = templates[1]
h,w = center_point(template)

with mss() as sct:
    while True:
        image = np.array(sct.grab(search_area))
        width, height = image.shape[1], image.shape[0]
        image_gray = cv.cvtColor(np.array(image), cv.COLOR_RGB2GRAY)

        print("Template:", template.shape)
        print("Screen:", image_gray.shape)
        
        max_val, max_loc = find_template(image_gray, template)
        
        # cv.imshow("screen", image_gray)
        # cv.imshow("template", template)
        # cv.waitKey(0)
        time.sleep(0.1)
        
        image_second = np.array(sct.grab(search_area))
        width, height = image_second.shape[1], image_second.shape[0]
        image_second_gray = cv.cvtColor(np.array(image_second), cv.COLOR_RGB2GRAY)

        max_val_second, max_loc_second = find_template(image_second_gray, template, threshold=0.8)


        print(f"Maximální hodnota shody: {max_val:.2f} na pozici {max_loc}.")

        if max_val >= 0.5 and max_val_second >= 0.5:            
            roi, offset = get_roi(image_second_gray, max_loc, template.shape, margin=100)
            
            res_2 = cv.matchTemplate(roi, template, cv.TM_CCOEFF_NORMED)
            max_val_second, max_loc_roi = find_template(roi, template, threshold=0.8)
            
            # Přepočti souřadnice z ROI zpět na celou obrazovku            
            if max_val_second >= 0.8:
                max_loc_second = (max_loc_roi[0] + offset[0], max_loc_roi[1] + offset[1])
                fail_count = 0 
                if show_img_result:
                    cv.rectangle(image, max_loc, (max_loc[0] + template.shape[1], max_loc[1] + template.shape[0]), (0,255,0), 2)
                    cv.rectangle(image, max_loc_second, (max_loc_second[0] + template.shape[1], max_loc_second[1] + template.shape[0]), (0,0,255), 2)

                # Teď už x1, y1 a x2, y2 patří stejnému terči!
                x1, y1 = max_loc[0] + w, max_loc[1] + h
                x2, y2 = max_loc_second[0] + w, max_loc_second[1] + h

                vx, vy = x2 - x1, y2 - y1
                
                # Predikce
                pred_x, pred_y = int(x2 + vx), int(y2 + vy)
                
                click_on_target(pred_x + mon_width, pred_y)
                
                if show_img_result:
                    cv.circle(image, (x1, y1), 6, (0, 255, 0), -1)
                    cv.circle(image, (x2, y2), 6, (0, 0, 255), -1)
                    cv.circle(image, (pred_x, pred_y), 6, (255, 0, 0), -1)
        else:
            print("Nenalezen žádný target.")
            fail_count += 1

        if fail_count >= cnt_rety:

            max_val, max_loc = find_template(image_gray, restrat, threshold=0.8)
            print(f"Maximální hodnota shody: {max_val:.2f} na pozici {max_loc}.")
            
            if max_val >= 0.8:
                fail_count = 0
                if show_img_result:
                    cv.rectangle(image, max_loc, (max_loc[0] + restrat.shape[1], max_loc[1] + restrat.shape[0]), (0, 255, 0), 2)
                x = max_loc[0] + restrat_w + mon_width
                y = max_loc[1] + restrat_h
                click_on_target(x, y)

        # Zobrazení obrázku v OpenCV
        if show_img_result:
            save_pic(path_debug, image)
        
        time.sleep(0.5)