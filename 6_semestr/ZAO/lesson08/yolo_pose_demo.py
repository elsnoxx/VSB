import os
import numpy as np
import cv2
from ultralytics import YOLO


def calculate_angle(a, b, c):
    a = np.array(a)  # hip
    b = np.array(b)  # knee
    c = np.array(c)  # ankle

    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cosine))

    return angle


def loadFiles(path):
    files = []
    labels = []
    for file in sorted(os.listdir(path)):
        full = os.path.join(path, file)
        files.append(full)
        if file.endswith(".mp4"):
            number = int(file.split(".")[0].split("_")[1].replace("d", ""))
            labels.append(number)
    return files, labels

def calculateAccurency(video_path, up_thresh, down_thresh):
    cap = cv2.VideoCapture(video_path)
    counter = 0
    stage = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("cap.read() chyba")
            break

        results = model(frame, verbose=False)

        # Výchozí nakreslení kostry od Ultralytics
        annotated_frame = results[0].plot()

        # Výpis souřadnic a vykreslení pomocí OpenCV
        if results[0].keypoints is not None:
            keypoints = results[0].keypoints.xy.cpu().numpy()
            # získat i confidence (jistotu) lze například takto:
            # keypoints_conf = results[0].keypoints.conf.cpu().numpy()   # jen hodnoty od 0 do 1
            
            for person_idx, pts in enumerate(keypoints):
                # Ukázka jak získat a vypsat konkrétní bod
                # V proměnné 'pts' je 17 bodů. Pomocí indexu [0] získáme první z nich - nos.
                nos_x, nos_y = pts[0]

                hip = pts[12]
                knee = pts[14]
                ankle = pts[16]

                angle = calculate_angle(hip, knee, ankle)
                if np.isnan(angle):
                    continue

                # přejít do "down" když se ohne (z up nebo None)
                if angle < down_thresh and stage != "down":
                    stage = "down"

                # když se vrátí nahoru z "down", přičti opakování
                if angle > up_thresh and stage == "down":
                    stage = "up"
                    counter += 1
                    print(f"Dřep: {counter}")
                
                if nos_x > 0: # 0,0 znamená, že síť bod nenašla / nevidí
                    print(f"Osoba {person_idx} -> Nos (bod 0) se nachází na X: {nos_x:.0f}, Y: {nos_y:.0f}")
                
                # OpenCV vykreslení do obrazu (body a čísla)
                for i, (x, y) in enumerate(pts):
                    if x > 0 and y > 0:
                        px, py = int(x), int(y)
                        # pozice klíčového bodu (žlutá)
                        cv2.circle(annotated_frame, (px, py), 4, (0, 255, 255), -1)
                        
                        # Vykreslení čísla klíčového bodu
                        cv2.putText(annotated_frame, str(i), (px + 5, py - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA) # Černé pozadí
                        cv2.putText(annotated_frame, str(i), (px + 5, py - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA) # Bílý text
                        # Vykreslení spojů mezi klíčovými body (modrá)
                        cv2.putText(annotated_frame, f"Angle: {int(angle)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                        cv2.putText(annotated_frame, f"Squats: {counter}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        cv2.imshow(f"YOLO Pose {video_path}", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return counter


if __name__ == "__main__":

    model = YOLO('yolo26n-pose.pt') 

    files, labels = loadFiles("dataset_analyza_pohybu")

    # počítadlo dřepů
    up_thresh = 150
    down_thresh = 90
    final_result = 0

    for i in range(len(labels)):
        result = calculateAccurency(files[i], up_thresh, down_thresh)

        if result == labels[i]:
            final_result += 1
            print(f"Video {i} -> OK")
        else:
            print(f"Video {i} -> Chyba, očekáváno: {labels[i]}, získáno: {result}")

    print(f"Celková přesnost: {final_result}/{len(labels)} = {final_result/len(labels)*100:.2f}%")

