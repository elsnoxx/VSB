import cv2

def load_templates():
    results = []
    with open("eye-state.txt", "r") as f:
        for line in f:
            clean_text = line.strip()
            if clean_text:
                results.append(clean_text)
            
    return results

def load_frontal():
    return cv2.CascadeClassifier(
        "haarcascades/haarcascades/haarcascade_frontalface_default.xml"
    )

def load_profile():
    return cv2.CascadeClassifier(
        "haarcascades/haarcascades/haarcascade_profileface.xml"
    )
    
def load_gescure():
    return cv2.CascadeClassifier(
        "eye_cascade_fusek/eye_cascade_fusek.xml"
    )

def load_video():
    return cv2.VideoCapture("fusek_face_car_01.avi")

def calculate_accurency(predictions, results):
    correct_count = 0
    empty = 0
    cnt = 0
    for p, gt in zip(predictions, results):
        if p == gt:
            correct_count += 1
        if p == "close":
            empty+= 1
        print(f"Snímek {cnt}: Predikce='{p}', GroundTruth='{gt}'")
        cnt+=1

    accuracy = (correct_count / len(predictions)) * 100
    print(f"\n--- VÝSLEDKY ---")
    print(f"Celkem snímků: {len(predictions)}")
    print(f"Počet predikci: {len(predictions)} Počet vstupnich dat: {len(results)}")
    print(f"Správných predikcí: {correct_count}")
    print(f"Prazdnych snimku: {empty}")
    print(f"Accuracy: {accuracy:.2f} %")