from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Read image
img = cv2.imread("fire.jpg")
if img is None:
    print("Error: 'fire.jpg' not found!")
    exit()

# Detect objects
results = model(img)

# Color for fire
color = (0, 0, 255)

# Draw boxes for fire only
for result in results:
    if result.boxes is not None:
        for box, cls_id in zip(result.boxes.xyxy, result.boxes.cls):
            cls_id = int(cls_id)
            label = result.names[cls_id]
            if label.lower() == "fire":
                x1, y1, x2, y2 = map(int, box.cpu().numpy())
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label.upper(), (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

# Show the result
cv2.imshow("Fire Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
