from mss import mss
import cv2 as cv
import numpy as np
import pyautogui
import time

#big screen
scale_factor = 4.5
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

prev_x, prev_y = None, None
cnt = 0

with mss() as sct:
    while True:
        image = np.array(sct.grab(search_area))
        width, height = image.shape[1], image.shape[0]
        image_gray = cv.cvtColor(np.array(image), cv.COLOR_RGB2GRAY)

        print("Template:", template.shape)
        print("Screen:", image_gray.shape)
        # cv.imshow("screen", image_gray)
        # cv.imshow("template", template)
        # cv.waitKey(0)


        res = cv.matchTemplate(image_gray, template, cv.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv.minMaxLoc(res)


        print(f"Maximální hodnota shody: {max_val:.2f} na pozici {max_loc}.")

        if max_val >= 0.8:
            if show_img_result:
                cv.rectangle(image, max_loc, (max_loc[0] + template.shape[1], max_loc[1] + template.shape[0]), (0, 255, 0), 2)
            # aktuální pozice středu objektu
            x = max_loc[0] + w + mon_width
            y = max_loc[1] + h

            if prev_x is not None:
                dist = np.hypot(x - prev_x, y - prev_y)

                if dist > 100:   # nový target
                    print("Nový target - reset predikce")
                    prev_x, prev_y = None, None
                    click_on_target(x, y)

                else:
                    vx = x - prev_x
                    vy = y - prev_y

                    pred_x = int(x + vx)
                    pred_y = int(y + vy)

                    click_on_target(pred_x, pred_y)

            else:
                # první detekce → ještě nemáme rychlost
                click_on_target(x, y)

            # uložíme aktuální pozici
            prev_x, prev_y = x, y
        else:
            print("Nenalezen žádný target.")
            fail_count += 1

        # if cv.waitKey(1) & 0xFF == ord('q'):
        #     break
        if fail_count >= 5:
            fail_count = 0
            res = cv.matchTemplate(image_gray, play, cv.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv.minMaxLoc(res)
            print(f"Maximální hodnota shody: {max_val:.2f} na pozici {max_loc}.")
            
            if max_val >= 0.8:
                if show_img_result:
                    cv.rectangle(image, max_loc, (max_loc[0] + play.shape[1], max_loc[1] + play.shape[0]), (0, 255, 0), 2)
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
                image = cv.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
                cv.imshow('Snímek obrazovky', image)
                cv.waitKey(0)
                cv.destroyAllWindows()
        
        time.sleep(0.5)