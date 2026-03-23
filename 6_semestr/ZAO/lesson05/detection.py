import cv2

frontal_scaleFactor = 1.2
frontal_minNeighbors = 10
profile_scaleFactor = 1.05
profile_minNeighbors = 5

def detect_eye_state(eye_roi):
    # 1. Agresivnější oříznutí (vystřihneme jen střed oka)
    # Tím se zbavíme obočí (nahoře) a kůže kolem (po stranách)
    h, w = eye_roi.shape
    # Ořízneme: shora 35%, zespodu 15%, zleva 15%, zprava 15%
    roi = eye_roi[int(h*0.35):int(h*0.85), int(w*0.15):int(w*0.85)]
    
    if roi.size == 0:
        return "close", 0

    # 2. Předzpracování
    roi = cv2.equalizeHist(roi)  # Srovnáme kontrast (důležité v autě)
    blur = cv2.GaussianBlur(roi, (5, 5), 0)

    # 3. OTSU Thresholding (automaticky najde hranici mezi zornicí a víčkem)
    # THRESH_BINARY_INV udělá z tmavé zornice bílé pixely
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 4. Morfologické operace (vymaže řasy a drobné šumy)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.erode(thresh, kernel, iterations=1)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    # Ukážeme si, co algoritmus vidí (pro ladění)
    # cv2.imshow("Debug Eye Thresh", cv2.resize(thresh, (150, 150)))

    # 5. Výpočet poměru "zornice" k ploše
    white_pixels = cv2.countNonZero(thresh)
    total_pixels = thresh.shape[0] * thresh.shape[1]
    ratio = white_pixels / total_pixels

    # Experimentální práh pro OTSU: 
    # Pokud je v oříznutém středu víc než 5-10 % bílé (zornice), je oko otevřené
    if ratio > 0.08: 
        return "open", ratio
    else:
        return "close", ratio
    
def detec_frontal_face(frame, face_frontal):
    rects_f, _, weights_f = face_frontal.detectMultiScale3(frame, frontal_scaleFactor, frontal_minNeighbors, outputRejectLevels=True)
    return rects_f, weights_f


def detec_profile_face(frame, face_profile):
    rects_f, _, weights_f = face_profile.detectMultiScale3(frame, profile_scaleFactor, profile_minNeighbors, outputRejectLevels=True)
    return rects_f, weights_f