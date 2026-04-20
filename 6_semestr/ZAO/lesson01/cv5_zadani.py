"""
PŘÍPRAVA PROSTŘEDÍ:
* Instalace balíku Ultralytics: 'pip install ultralytics' 
* https://github.com/ultralytics/ultralytics
* https://docs.ultralytics.com/modes/predict/#inference-arguments

Zadání na cvičení:

1. Inicializace detekčního modelu YOLO.
2. Načtení všech obrazových souborů z adresáře "bmw_100" a spuštění modelu YOLO na těchto souborech.
3. Implementace parametrů příkazové řádky: 
   - pro definici ID detekované třídy (např. 2 pro automobily).
   - pro velikost modelu
   - pro nazev vstupniho adresare
   - pro nazev vystupniho adresare
4. Uložení extrahovaných výřezů objektů do určené složky (např. 'car').
5. Vykreslení ohraničujících rámečků kolem objektu do původních obrazů (využití OpenCV). 
   Pokuste se o vykreslení ohraničujících rámečků (bounding boxes), které vrátí YOLO model pomocí funkce cv2.rectangle() případně informací o objektech + cv2.putText()
   - možnost nastavit barvu jako parametr příkazové řádky 
6. Vložení textové informace o celkovém počtu detekcí dané třídy do obrazu (např. levý dolní roh).

* ukázka spuštění s definicí barvy a specifické třídy:
python cv5_zadani.py --class_id 0 --color 0 0 255 --output_dir persons

0	osoba
1	jízdní kolo
2	osobní automobil
3	motocykl
5	autobus
"""
import os
import argparse
import cv2 as cv
from ultralytics import YOLO

# Inicializace modelu a cest
# ...

pareser = argparse.ArgumentParser(description="Detekce objektů pomocí YOLO modelu")
pareser.add_argument("--class_id", type=int, default=2, help="ID detekované třídy")
pareser.add_argument("--color", nargs=3, type=int, default=[0, 0, 255], help="Barva ohraničujících rámečků (B G R)")
pareser.add_argument("--input_dir", type=str, default="bmw_100", help="Název vstupního adresáře")
pareser.add_argument("--output_dir", type=str, default="output", help="Název výstupního adresáře")
args = pareser.parse_args()

files = []

path = args.input_dir
for file in os.listdir(path):
    # print(file)
    files.append(os.path.join(path, file))

model = YOLO("yolo26n.pt")

if(os.path.exists(args.output_dir) == False):
    os.mkdir(args.output_dir)

if(os.path.exists(os.path.join(args.output_dir, "croped")) == False):
    os.mkdir(os.path.join(args.output_dir, "croped"))

# Iterace souborů a čtení obrazů
# ...

data = {}

objects = []
for file in files:
    img = cv.imread(file)
    prediction = model(img, device="cpu", show=False, classes=[args.class_id])
    objectDetected = prediction[0].boxes
    data[file] = objectDetected



# Inference, extrakce ROI a zápis na disk
# ...    
cnt = 0
img_counter = 0
for item in data:
    cnt = 0
    boxes = data[item]
    img = cv.imread(item)
    # print(item)
    for box in boxes:
        # print(box)
        x1, y1, x2, y2 = box.xyxy[0]
        conf = box.conf[0]
        cls = box.cls[0]
        if conf < 0.3:
            continue
        cropt_img = img[int(y1):int(y2), int(x1):int(x2)]
        cv.imwrite(os.path.join(args.output_dir, "croped" ,f"{os.path.basename(item).split('.')[0]}_crop{cnt}.jpg"), cropt_img)
        cnt += 1
        # print(f"Object: x1={x1}, y1={y1}, x2={x2}, y2={y2}, confidence={conf}")
        cv.putText(img, f"Class: {cls}, Conf: {conf:.2f}", (int(x1), int(y1) - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, tuple(args.color), 2)
        cv.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), tuple(args.color), 3)
    cv.putText(img, f"Total Detected: {len(boxes)}", (10, img.shape[0] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, tuple(args.color), 2)
    cv.imshow('Custom drawing', img)
    cv.imwrite(os.path.join(args.output_dir, f"{os.path.basename(files[img_counter]).split('.')[0]}_detection.jpg"), img)
    cv.waitKey(0)
    img_counter += 1
    
