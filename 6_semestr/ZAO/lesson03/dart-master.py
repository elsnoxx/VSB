from mss import mss
import cv2 as cv
import numpy as np
import pyautogui
import time
import os

#big screen
scale_factor = 4.6
# scale_factor = 6

scale_factor_other = 1

show_img_result = False
fail_count = 0

def click_on_target(x, y):
    pyautogui.click(x, y, _pause=False)
    print(f"Klikám... {x} a {y}")

template = cv.imread('dartMaster/target.png', 0)
template = cv.resize(template, (int(template.shape[1] // scale_factor), int(template.shape[0] // scale_factor)))


play = cv.imread('dartMaster/play.png', 0)
play = cv.resize(play, (int(play.shape[1] // scale_factor_other), int(play.shape[0] // scale_factor_other)))

restrat = cv.imread('dartMaster/restart.png', 0)
restrat = cv.resize(restrat, (int(restrat.shape[1] // scale_factor_other), int(restrat.shape[0] // scale_factor_other)))


h, w = template.shape[:2]
w = w // 2
h = h // 2

play_h, play_w = play.shape[:2]
play_w = play_w // 2
play_h = play_h // 2

restrat_h, restrat_w = restrat.shape[:2]
restrat_w = restrat_w // 2
restrat_h = restrat_h // 2

if template is None:
    print("Chyba: Soubor target.png nebyl nalezen!")
    exit()

mon_width, mon_height = pyautogui.size()
mon_width = mon_width // 2

search_area = {"top": 0, "left": mon_width, "width": mon_width, "height": mon_height}

cnt = 0

with mss() as sct:
    while True:
        image = np.array(sct.grab(search_area))
        width, height = image.shape[1], image.shape[0]
        image_gray = cv.cvtColor(np.array(image), cv.COLOR_RGB2GRAY)

        print("Template:", template.shape)
        print("Screen:", image_gray.shape)
        
        res = cv.matchTemplate(image_gray, template, cv.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv.minMaxLoc(res)
        
        # cv.imshow("screen", image_gray)
        # cv.imshow("template", template)
        # cv.waitKey(0)
        time.sleep(0.1)
        
        image_second = np.array(sct.grab(search_area))
        width, height = image_second.shape[1], image_second.shape[0]
        image_second_gray = cv.cvtColor(np.array(image_second), cv.COLOR_RGB2GRAY)


        res_2 = cv.matchTemplate(image_second_gray, template, cv.TM_CCOEFF_NORMED)
        _, max_val_second, _, max_loc_second = cv.minMaxLoc(res_2)


        print(f"Maximální hodnota shody: {max_val:.2f} na pozici {max_loc}.")

        if max_val >= 0.8 and max_val_second >= 0.8:
            # Definuj oblast kolem prvního nálezu (např. + - 100 pixelů)
            margin = 100
            top = max(0, max_loc[1] - margin)
            bottom = min(image_gray.shape[0], max_loc[1] + h*2 + margin)
            left = max(0, max_loc[0] - margin)
            right = min(image_gray.shape[1], max_loc[0] + w*2 + margin)

            # Druhý grab udělej klidně celé obrazovky, ale matchuj jen v ROI
            roi = image_second_gray[top:bottom, left:right]
            res_2 = cv.matchTemplate(roi, template, cv.TM_CCOEFF_NORMED)
            _, max_val_second, _, max_loc_roi = cv.minMaxLoc(res_2)
            
            # Přepočti souřadnice z ROI zpět na celou obrazovku
            max_loc_second = (max_loc_roi[0] + left, max_loc_roi[1] + top)
            
            if max_val_second >= 0.8:
            
                fail_count = 0 
                if show_img_result:
                    cv.rectangle(image, max_loc, (max_loc[0] + template.shape[1], max_loc[1] + template.shape[0]), (0,255,0), 2)
                    cv.rectangle(image, max_loc_second, (max_loc_second[0] + template.shape[1], max_loc_second[1] + template.shape[0]), (0,0,255), 2)
                # 3. Přepočet souřadnic z výřezu zpět na celou plochu
                max_loc_second = (max_loc_roi[0] + left, max_loc_roi[1] + top)

                # Teď už x1, y1 a x2, y2 patří stejnému terči!
                x1, y1 = max_loc[0] + w, max_loc[1] + h
                x2, y2 = max_loc_second[0] + w, max_loc_second[1] + h

                vx, vy = x2 - x1, y2 - y1
                
                # Predikce (můžeš vektor i vynásobit pro větší předstih)
                pred_x, pred_y = int(x2 + vx), int(y2 + vy)
                
                click_on_target(pred_x + mon_width, pred_y)
                
                if show_img_result:
                    cv.circle(image, (x1, y1), 6, (0, 255, 0), -1)
                    cv.circle(image, (x2, y2), 6, (0, 0, 255), -1)
                    cv.circle(image, (pred_x, pred_y), 6, (255, 0, 0), -1)
        else:
            print("Nenalezen žádný target.")
            fail_count += 1

        if fail_count >= 10:
            fail_count = 0
            res = cv.matchTemplate(image_gray, play, cv.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv.minMaxLoc(res)
            print(f"Maximální hodnota shody: {max_val:.2f} na pozici {max_loc}.")
            
            if max_val >= 0.8:
                if show_img_result:
                    cv.rectangle(image, max_loc, (max_loc[0] + play.shape[1], max_loc[1] + play.shape[0]), (0, 255, 0), 2)
                    cv.rectangle(image_second, max_loc_second, (max_loc_second[0] + play.shape[1], max_loc_second[1] + play.shape[0]), (255, 0, 0), 2)
                x = max_loc[0] + play_w + mon_width
                y = max_loc[1] + play_h
                click_on_target(x, y)

            res = cv.matchTemplate(image_gray, restrat, cv.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv.minMaxLoc(res)
            print(f"Maximální hodnota shody: {max_val:.2f} na pozici {max_loc}.")
            
            if max_val >= 0.8:
                if show_img_result:
                    cv.rectangle(image, max_loc, (max_loc[0] + restrat.shape[1], max_loc[1] + restrat.shape[0]), (0, 255, 0), 2)
                x = max_loc[0] + restrat_w + mon_width
                y = max_loc[1] + restrat_h
                click_on_target(x, y)

        # Zobrazení obrázku v OpenCV
        if show_img_result:
            path = "debug"
            if(os.path.exists(path) == False):
                os.mkdir(path)
            result = cv.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
            filename = time.strftime("%Y%m%d_%H%M%S") + ".jpg"
            cv.imwrite(os.path.join(path ,filename), image)
            
            # cv.imshow('Snímek obrazovky', image)
            # cv.waitKey(0)
            # cv.destroyAllWindows()
        
        time.sleep(0.5)