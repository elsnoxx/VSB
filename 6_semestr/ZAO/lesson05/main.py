import cv2
import helpers
import detection
import drawings


results = helpers.load_templates()
face_frontal = helpers.load_frontal()
face_profile = helpers.load_profile()
face_gescure = helpers.load_gescure()
cap = helpers.load_video()


predictions = []
while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Pomocná proměnná pro stav v tomto snímku (výchozí je 'close', pokud nic nenajdem)
    current_frame_state = "close" 

    start_time = cv2.getTickCount()

    # 1. Nejdříve zkusíme Frontal
    rects_f, weights_f = detection.detec_frontal_face(frame, face_frontal)
    
    faces_to_process = []

    # Pokud jsme našli frontální obličej s dostatečnou vahou
    found_frontal = False
    for i, f in enumerate(rects_f):
        if weights_f[i] > 2.0:
            faces_to_process.append((f, "Frontal", (255, 0, 0)))
            found_frontal = True
            break # Pro řidiče v autě nám stačí jeden obličej

    # 2. POUZE POKUD jsme nenašli Frontal, zkusíme Profile
    if not found_frontal:
        rects_p, weights_p = detection.detec_profile_face(frame, face_profile)
        for i, f in enumerate(rects_p):
            if weights_p[i] > 2.0:
                faces_to_process.append((f, "Profile", (0, 255, 0)))
                break

    end_time = cv2.getTickCount()
    time_ms = (end_time - start_time) / cv2.getTickFrequency() * 1000

    # B. DETEKCE OČÍ UVNITŘ OBLIČEJE
    for (box, name, color) in faces_to_process:
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Vytvoříme ROI (Region of Interest) - výřez pouze obličeje
        face_roi_gray = gray[y:y+h, x:x+w]
        
        # Hledáme oči jen v horní polovině obličeje (zrychlení a přesnost)
        eyes_roi = face_roi_gray[0:int(h*0.6), :]
        
        rects_g, _, weights_g = face_gescure.detectMultiScale3(
            eyes_roi, scaleFactor=1.1, minNeighbors=5, outputRejectLevels=True
        )

        for i, (ex, ey, ew, eh) in enumerate(rects_g):
            if weights_g[i] > 1.5:
                # Výřez konkrétního oka pro analýzu stavu (open/close)
                eye_final_roi = eyes_roi[ey:ey+eh, ex:ex+ew]
                
                state, ratio = detection.detect_eye_state(eye_final_roi)
                current_frame_state = state # Uložíme výsledek

                # Vykreslení oka
                drawings.draw_eye(frame,x,y,ex,ey,ew,eh,state)
                
                break

    # Uložíme predikci pro výpočet accuracy na konci
    predictions.append(current_frame_state)
    # Info display
    drawings.info_display(frame, time_ms)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
helpers.calculate_accurency(predictions, results)