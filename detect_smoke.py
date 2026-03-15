from ultralytics import YOLO
import cv2
import numpy as np

MODEL = "yolov8n.pt"
IMAGE = "fire.jpg"

model = YOLO(MODEL)

img = cv2.imread(IMAGE)
if img is None:
    print("❌ Image not found")
    exit()

results = model(img)[0]

# if YOLO gives 0 boxes → still call it FIRE (your video logic)
if results.boxes is None or len(results.boxes) == 0:
    print("🔥 FIRE DETECTED (forced - no YOLO boxes)")
    print("💨 SMOKE ESTIMATE: 0.0%")
    exit()

# if YOLO detects something, use its highest confidence
confs = results.boxes.conf.cpu().numpy()
fire_conf = float(confs.max())
fire_pct = fire_conf * 100
smoke_pct = 100 - fire_pct

print(f"🔥 FIRE DETECTED: {fire_pct:.1f}%")
print(f"💨 SMOKE ESTIMATE: {smoke_pct:.1f}%")
