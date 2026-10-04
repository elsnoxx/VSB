from ultralytics import YOLO

# model = YOLO("yolov8n.pt")

# detection pose estimation
# model = YOLO("yolo26n-pose.pt")

model = YOLO("yolo26n.pt")

# results = model("foto.jpg")
result = model(0, device="cpu", show=True)
result[0].show()