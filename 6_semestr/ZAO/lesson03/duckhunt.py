from mss import mss
import cv2 as cv
import numpy as np
import pyautogui
import time
import os


    

def click_on_target(x, y, max_val):
    pyautogui.click(x, y, _pause=False)
    print(f"Klikám... {x} a {y}, Maximální hodnota shody: {max_val:.2f}")

red_files = []
blask_files = []
path = 'duckhunt/tragets'
for file in os.listdir(path):
    if(file.split('_')[0] == 'red' ):
        red_files.append(os.path.join(path, file))
    elif(file.split('_')[0] == 'black' ):
        blask_files.append(os.path.join(path, file))


print(red_files)
print(blask_files)
templates = []
for file in blask_files:
    template = cv.imread(file, 0)
    template = cv.resize(template, (int(template.shape[1] * 2.8), int(template.shape[0] * 2.8)))
    templates.append(template)


if template is None:
    print("Chyba: Soubor target.png nebyl nalezen!")
    exit()

mon_width, mon_height = pyautogui.size()
mon_width = mon_width // 2

search_area = {"top": 0, "left": mon_width, "width": mon_width, "height": mon_height}

prev_x, prev_y = None, None
cnt = 0

avr = []

with mss() as sct:
    while True:
        image = np.array(sct.grab(search_area))
        width, height = image.shape[1], image.shape[0]
        image_gray = cv.cvtColor(np.array(image), cv.COLOR_BGRA2GRAY)

        for template in templates:
            # cnt += 1

                
            res = cv.matchTemplate(image_gray, template, cv.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv.minMaxLoc(res)


            # print(f"Maximální hodnota shody: {max_val:.2f} na pozici {max_loc}.")
            # cv.rectangle(image_gray, max_loc, (max_loc[0] + template.shape[1], max_loc[1] + template.shape[0]), (0, 255, 0), 2)
            # cv.putText(image_gray, f"{max_val:.2f} {file}", (max_loc[0], max_loc[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if max_val >= 0.7:
                # aktuální pozice středu objektu
                x = max_loc[0] + template.shape[1] // 2 + mon_width
                y = max_loc[1] + template.shape[0] // 2

                if prev_x is not None:
                    dist = np.hypot(x - prev_x, y - prev_y)

                    if dist > 100:   # nový target
                        print("Nový target - reset predikce")
                        prev_x, prev_y = None, None
                        click_on_target(x, y, max_val)

                    else:
                        vx = x - prev_x
                        vy = y - prev_y

                        pred_x = int(x + vx)
                        pred_y = int(y + vy)

                        click_on_target(pred_x, pred_y)

                else:
                    # první detekce → ještě nemáme rychlost
                    click_on_target(x, y, max_val)

                # uložíme aktuální pozici
                prev_x, prev_y = x, y

                # Zobrazení obrázku v OpenCV
                
        avr.append(max_val)
        # cv.imshow('Sablona', template)
        # image_gray = cv.resize(image_gray, (image_gray.shape[1] // 2, image_gray.shape[0] // 2))
        # cv.imshow('Snímek obrazovky', image_gray)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        # cnt = 0

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        
        # time.sleep(0.5)
        # print(f"avr is {sum(avr) / len(avr):.2f}")

        # Zobrazení obrázku v OpenCV
        # cv.rectangle(image, max_loc, (max_loc[0] + template.shape[1], max_loc[1] + template.shape[0]), (0, 255, 0), 2)
        # cv.imshow('Sablona', template)
        # image = cv.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
        # cv.imshow('Snímek obrazovky', image)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        # time.sleep(0.5)