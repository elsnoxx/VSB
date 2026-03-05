from mss import mss
import cv2 as cv
import numpy as np
import pyautogui
import time

def click_on_target(x, y):
    pyautogui.click(x, y, _pause=False)
    print(f"Klikám... {x} a {y}")

template = cv.imread('target.png', 0)
template = cv.resize(template, (int(template.shape[1] // 4.5), int(template.shape[0] // 4.5)))
h, w = template.shape[:2]
w = w // 2
h = h // 2

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


        res = cv.matchTemplate(image_gray, template, cv.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv.minMaxLoc(res)


        print(f"Maximální hodnota shody: {max_val:.2f} na pozici {max_loc}.")

        if max_val >= 0.8:
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

        if cv.waitKey(1) & 0xFF == ord('q'):
                break
        
        time.sleep(0.5)

        # Zobrazení obrázku v OpenCV
        # cv.rectangle(image, max_loc, (max_loc[0] + template.shape[1], max_loc[1] + template.shape[0]), (0, 255, 0), 2)
        # cv.imshow('Sablona', template)
        # image = cv.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
        # cv.imshow('Snímek obrazovky', image)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        # time.sleep(0.5)