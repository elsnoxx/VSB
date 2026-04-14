from collections import deque
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

def calculateAccurency(video_path, up_thresh, down_thresh, drop_up_limit, drop_down_limit, img_show="show"):
    cap = cv2.VideoCapture(video_path)
    counter = 0
    angle_history = deque(maxlen=5)
    stage = None
    initial_leg_height = None
    calibration_frames = 0

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
            keypoints_conf = results[0].keypoints.conf.cpu().numpy()   # jen hodnoty od 0 do 1
            
            for person_idx, pts in enumerate(keypoints):
                # Ukázka jak získat a vypsat konkrétní bod
                # V proměnné 'pts' je 17 bodů. Pomocí indexu [0] získáme první z nich - nos.
                nos_x, nos_y = pts[0]

                # Validace bodů (např. confidence > 0.5)
                l_angle = calculate_angle(pts[11], pts[13], pts[15])
                r_angle = calculate_angle(pts[12], pts[14], pts[16])
                
                # Vybereme nejlepší dostupný úhel
                current_angle = None
                if l_angle and r_angle:
                    current_angle = (l_angle + r_angle) / 2
                elif l_angle:
                    current_angle = l_angle
                elif r_angle:
                    current_angle = r_angle

                if current_angle is not None:
                    angle_history.append(current_angle)
                    # Vyhlazený úhel (průměr z historie)
                    smooth_angle = sum(angle_history) / len(angle_history)
                    # 2. Výpočet relativního poklesu (pro čelní pohled)
                    # Průměrná Y souřadnice kyčlí a kotníků
                    avg_hip_y = (pts[11][1] + pts[12][1]) / 2
                    avg_ankle_y = (pts[15][1] + pts[16][1]) / 2
                    current_leg_height = abs(avg_ankle_y - avg_hip_y)

                    # Kalibrace na začátku (předpokládáme, že člověk první 5-10 snímků stojí)
                    if current_leg_height > 0 and calibration_frames < 10:
                        if initial_leg_height is None: initial_leg_height = current_leg_height
                        else: initial_leg_height

                    drop_ratio = current_leg_height / initial_leg_height if initial_leg_height else 1.0
                    print(f"Smooth angle: {smooth_angle}, Drop ratio: {drop_ratio}")
                    # Logika počítání s vyhlazeným úhlem
                    is_down = smooth_angle < down_thresh or drop_ratio < drop_down_limit
                    is_up = smooth_angle > up_thresh and drop_ratio > drop_up_limit

                    if is_down and stage != "down":
                        stage = "down"
                    
                    if is_up and stage == "down":
                        stage = "up"
                        counter += 1
                
                if nos_x > 0: # 0,0 znamená, že síť bod nenašla / nevidí
                    print(f"Osoba {person_idx} -> Nos (bod 0) se nachází na X: {nos_x:.0f}, Y: {nos_y:.0f}")
                
                # OpenCV vykreslení do obrazu (body a čísla)
                if img_show == "show":
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
                            cv2.putText(annotated_frame, f"Angle: {int(current_angle)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                            cv2.putText(annotated_frame, f"Squats: {counter}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                            cv2.putText(annotated_frame, f"Drop Ratio: {drop_ratio:.2f}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        if img_show == "show":
            cv2.imshow(f"YOLO Pose {video_path}", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    return counter


if __name__ == "__main__":

    model = YOLO('yolo26n-pose.pt') 

    files, labels = loadFiles("dataset_analyza_pohybu")

    up_thresh = 100
    down_thresh = 90
    drop_up_limit = 0.7
    drop_down_limit = 0.7

    correct = 0
    results_log = []

    for i in range(len(labels)):
        predicted = calculateAccurency(files[i], up_thresh, down_thresh, drop_up_limit, drop_down_limit, img_show="show")

        if predicted == labels[i]:
            correct += 1
            results_log.append(f"Video {i+1} -> OK ({predicted})")
        else:
            results_log.append(
                f"Video {i+1} -> Chyba | očekáváno: {labels[i]}, získáno: {predicted}"
            )
        

    print("\n===== VÝSLEDKY =====")
    for res in results_log:
        print(res)

    accuracy = correct / len(labels) * 100
    print(f"\nCelková přesnost: {correct}/{len(labels)} = {accuracy:.2f}%")

