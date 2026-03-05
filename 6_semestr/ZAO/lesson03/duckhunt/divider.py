import cv2 as cv

sheet = cv.imread("sprites.png")

# 1) převod na šedou
gray = cv.cvtColor(sheet, cv.COLOR_BGR2GRAY)

# 2) vytvoření masky (černé pozadí pryč)
_, thresh = cv.threshold(gray, 10, 255, cv.THRESH_BINARY)

# 3) najdi kontury
contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

sprites = []

for c in contours:
    x, y, w, h = cv.boundingRect(c)

    # filtr: vynecháme moc malé objekty (šum)
    if w > 20 and h > 20:
        sprite = sheet[y:y+h, x:x+w]
        sprites.append(sprite)

        # volitelně ulož do souboru
        cv.imwrite(f"res/sprite_{x}_{y}.png", sprite)

print("nalezeno sprite:", len(sprites))