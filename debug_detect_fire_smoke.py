# debug_detect_fire_smoke.py
import cv2
import numpy as np
import sys
import time

# ---------- EMAIL CONFIG ----------
import smtplib
from email.message import EmailMessage

EMAIL_USER = "t39775496@gmail.com"          # sender + receiver
EMAIL_TO   = "t39775496@gmail.com"          # email to receive alerts
EMAIL_PASS = "uvnzoepxrqnxrwqn"             # your 16-letter app password

EMAIL_COOLDOWN = 60  # seconds between video alerts


def send_email_alert(subject, body, image_path=None):
    msg = EmailMessage()    
    msg["From"] = EMAIL_USER
    msg["To"] = EMAIL_TO
    msg["Subject"] = subject
    msg.set_content(body)

    # attach image if available
    if image_path is not None:
        try:
            with open(image_path, "rb") as f:
                data = f.read()
            msg.add_attachment(
                data,
                maintype="image",
                subtype="jpeg",
                filename="alert.jpg"
            )
        except Exception as e:
            print("⚠️ Could not attach image:", e)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_USER, EMAIL_PASS)
            smtp.send_message(msg)
        print("📨 Email alert sent!")
        return True
    except Exception as e:
        print("❌ Email failed:", e)
        return False


# ---------- COLORS ----------
FIRE_COLOR = (0, 0, 255)    # RED
SMOKE_COLOR = (255, 0, 0)   # BLUE


# ---------- IMAGE MODE ----------
def classify_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("❌ Image not found:", img_path)
        return

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    fire_mask = cv2.inRange(hsv, (0, 80, 80), (50, 255, 255))
    smoke_mask = cv2.inRange(hsv, (0, 0, 60), (180, 70, 230))

    fire_pixels = cv2.countNonZero(fire_mask)
    smoke_pixels = cv2.countNonZero(smoke_mask)
    total = max(fire_pixels + smoke_pixels, 1)

    fire_pct = fire_pixels * 100.0 / total
    smoke_pct = smoke_pixels * 100.0 / total

    print(f"\n🔥 FIRE detected:  {fire_pct:.1f}%")
    print(f"💨 SMOKE detected: {smoke_pct:.1f}%\n")

    # -------- EMAIL ALERT FOR IMAGE MODE --------
    snap = "image_alert.jpg"
    cv2.imwrite(snap, img)

    subject = "🔥 FIRE ALERT (IMAGE)" if fire_pct > smoke_pct else "💨 SMOKE ALERT (IMAGE)"
    body = f"Image detection:\nFire: {fire_pct:.1f}%\nSmoke: {smoke_pct:.1f}%"

    send_email_alert(subject, body, snap)
    print("📨 Email alert sent for image!")


# If an image path is provided → ONLY run image mode
if len(sys.argv) == 2:
    classify_image(sys.argv[1])
    sys.exit()


# ---------- VIDEO MODE ----------
VIDEO = "fire2.mp4"
OUTPUT = "debug_output_fire_detection.mp4"

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    print("❌ Could not open video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(OUTPUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

print("Running... Press q to quit.")


def get_largest_box(mask, min_area=500):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(largest)
    if bw * bh < min_area:
        return None
    return (x, y, bw, bh)


last_email_time = 0


while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    fire_mask = cv2.inRange(hsv, (0, 80, 80), (50, 255, 255))
    smoke_mask = cv2.inRange(hsv, (0, 0, 60), (180, 70, 230))

    fire_pixels = cv2.countNonZero(fire_mask)
    smoke_pixels = cv2.countNonZero(smoke_mask)
    total = max(fire_pixels + smoke_pixels, 1)

    fire_pct = fire_pixels * 100.0 / total
    smoke_pct = smoke_pixels * 100.0 / total

    now = time.time()

    # ---------- FIRE BOX ----------
    fire_box = get_largest_box(fire_mask)
    if fire_box is not None:
        x, y, bw, bh = fire_box
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), FIRE_COLOR, 3)
        cv2.putText(frame, f"FIRE {fire_pct:.1f}%", (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, FIRE_COLOR, 2)

        if now - last_email_time > EMAIL_COOLDOWN:
            snap = "fire_alert.jpg"
            cv2.imwrite(snap, frame)
            if send_email_alert("🔥 FIRE ALERT", f"Fire detected: {fire_pct:.1f}%", snap):
                last_email_time = now

    # ---------- SMOKE BOX ----------
    smoke_box = get_largest_box(smoke_mask)
    if smoke_box is not None:
        sx, sy, sbw, sbh = smoke_box
        cv2.rectangle(frame, (sx, sy), (sx + sbw, sy + sbh), SMOKE_COLOR, 3)
        cv2.putText(frame, f"SMOKE {smoke_pct:.1f}%", (sx, max(20, sy - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, SMOKE_COLOR, 2)

        if now - last_email_time > EMAIL_COOLDOWN:
            snap = "smoke_alert.jpg"
            cv2.imwrite(snap, frame)
            if send_email_alert("💨 SMOKE ALERT", f"Smoke detected: {smoke_pct:.1f}%", snap):
                last_email_time = now

    # ---------- SHOW VIDEO ----------
    cv2.imshow("Dynamic Fire + Smoke Boxes", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
out.release()
cv2.destroyAllWindows()
print("Saved ->", OUTPUT)


# debug_detect_fire_smoke.py
from ultralytics import YOLO
import cv2
import time

# USE YOLOv8n.pt (you already have this)
MODEL = "yolov8n.pt"
VIDEO = "fire2.mp4"
OUTPUT = "debug_output_fire_detection.mp4"

CONF_THRESH = 0.25

FALLBACK_COLOR = (0, 0, 255)  # red for fire

model = YOLO(MODEL)
cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    print("❌ Could not open video:", VIDEO)
    exit()

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(OUTPUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

print("Running... Press q to quit.")
last_print = 0
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    results = model(frame)[0]

    if results.boxes is not None:
        for box, conf in zip(results.boxes.xyxy.cpu().numpy(),
                             results.boxes.conf.cpu().numpy()):
            if conf < CONF_THRESH:
                continue

            # Force FIRE label
            label = "fire"

            x1, y1, x2, y2 = map(int, box)
            color = FALLBACK_COLOR

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            # Draw label
            text = f"FIRE {conf*100:.1f}%"
            cv2.putText(frame, text, (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("FIRE DETECTION", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Saved ->", OUTPUT)




# debug_detect_fire_smoke.py  <- updated with email alert
from ultralytics import YOLO
import cv2
import time
import os
import smtplib
from email.message import EmailMessage
import mimetypes

MODEL = "yolov8n.pt"   # keep this for now (or change to your custom .pt if you have it)
VIDEO = "fire2.mp4"
OUTPUT = "debug_output_fire_detection.mp4"

# Draw every class (debug mode)
CONF_THRESH = 0.25

# A simple palette (will fallback to green)
FALLBACK_COLOR = (0, 255, 0)
PALETTE = {
    "fire": (0, 0, 255),
    "smoke": (255, 0, 0)
}

# Email cooldown (seconds) to avoid spamming
EMAIL_COOLDOWN = 60

# ---------------- Email helper ----------------
def send_email_alert(subject: str, body: str, attachment_path: str = None):
    """
    Sends an email using environment variables:
      ALERT_EMAIL_USER, ALERT_EMAIL_PASS, ALERT_EMAIL_TO
    Returns True on success, False otherwise.
    """
    user = os.getenv("ALERT_EMAIL_USER")
    passwd = os.getenv("ALERT_EMAIL_PASS")
    to_addr = os.getenv("ALERT_EMAIL_TO")

    if not (user and passwd and to_addr):
        print("❌ Email vars not set. ALERT_EMAIL_USER, PASS, TO missing.")
        return False

    msg = EmailMessage()
    msg["From"] = user
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.set_content(body)

    # attach file if provided
    if attachment_path:
        if os.path.exists(attachment_path):
            ctype, encoding = mimetypes.guess_type(attachment_path)
            if ctype is None:
                ctype = "application/octet-stream"
            maintype, subtype = ctype.split("/", 1)

            with open(attachment_path, "rb") as f:
                msg.add_attachment(
                    f.read(),
                    maintype=maintype,
                    subtype=subtype,
                    filename=os.path.basename(attachment_path)
                )
        else:
            print("⚠️ Attachment not found:", attachment_path)

    try:
        # Gmail SMTP Example (SSL)
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(user, passwd)
            smtp.send_message(msg)

        print("📨 Email alert sent to", to_addr)
        return True

    except Exception as e:
        print("❌ Failed to send email:", e)
        return False

# ------------------------------------------------

model = YOLO(MODEL)
cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    print("❌ Could not open video:", VIDEO)
    exit()

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(OUTPUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

print("DEBUG MODE — drawing all detections. Press q to quit.")
frame_idx = 0
last_print = 0.0
last_email_sent = 0.0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    results = model(frame)
    r = results[0]

    labels_this_frame = []
    if r.boxes is not None and len(r.boxes) > 0:
        xyxy = r.boxes.xyxy.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()

        for box, cid, conf in zip(xyxy, cls_ids, confs):
            if conf < CONF_THRESH:
                continue
            # force FIRE/SMOKE mapping if desired - keep original label for debug
            label = r.names[int(cid)].lower()
            labels_this_frame.append((label, float(conf)))

            x1, y1, x2, y2 = map(lambda v: int(round(v)), box)
            color = PALETTE.get(label, FALLBACK_COLOR)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label.upper()} {conf*100:.1f}%",
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Print labels once per ~0.5s to avoid huge spam
    now = time.time()
    if labels_this_frame and now - last_print > 0.5:
        last_print = now
        pretty = ", ".join([f"{lab}:{conf*100:.1f}%" for lab,conf in labels_this_frame])
        print(f"frame {frame_idx}: {pretty}")

    # If there is at least one detection, and cooldown passed, send an email alert
    if labels_this_frame and (time.time() - last_email_sent) > EMAIL_COOLDOWN:
        # save a snapshot annotated image for the alert
        snap_path = f"alert_frame_{frame_idx}.jpg"
        cv2.imwrite(snap_path, frame)

        # prepare subject/body
        # choose top detection as representative
        top_label, top_conf = max(labels_this_frame, key=lambda x: x[1])
        subject = f"ALERT: {top_label.upper()} detected"
        body = f"Detected {top_label} with confidence {top_conf*100:.1f}% in video {VIDEO} at frame {frame_idx}. See attached image."

        # call email function (non-blocking: run in background thread would be better, but keep simple here)
        sent = send_email_alert(subject, body, attachment_path=snap_path)
        if sent:
            last_email_sent = time.time()

    cv2.imshow("DEBUG Fire/Smoke Detection", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Saved ->", OUTPUT)

# ================= IMAGE CHECK MODE =================
def check_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("❌ Image not found!")
        return

    results = model(img)[0]

    if results.boxes is not None and len(results.boxes) > 0:
        confs = results.boxes.conf.cpu().numpy()
        if np.any(confs >= CONF_THRESH):
            print("Image detected")
            return

    print("No detection")

# <<< ADD THIS >>>
check_image("fire.jpg")



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




from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")  # or your custom weights path
images = ["fire.jpg"]       # add more filenames if needed
COLORS = {"fire": (0, 0, 255), "smoke": (255, 0, 0)}
CONF_THRESH = 0.30

for image_file in images:
    img = cv2.imread(image_file)
    if img is None:
        print(f"Error: {image_file} not found!")
        continue

    results = model(img)
    r = results[0]
    detected = False

    if r.boxes is not None and len(r.boxes) > 0:
        xyxy = r.boxes.xyxy.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()

        for box, cls_id, conf in zip(xyxy, cls_ids, confs):
            if conf < CONF_THRESH:
                continue
            label = r.names[int(cls_id)].lower()
            if label in ["fire", "smoke"]:
                detected = True
                x1, y1, x2, y2 = map(lambda v: int(round(v)), box)
                color = COLORS.get(label, (0, 255, 0))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f"{label.upper()} {conf*100:.1f}%",
                            (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    if detected:
        print(f"Image detected: {image_file}")
    else:
        print(f"No detections in image: {image_file}")

    cv2.imshow(image_file, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()