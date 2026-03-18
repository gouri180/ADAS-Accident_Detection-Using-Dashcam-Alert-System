from ultralytics import YOLO
import cv2
import os
from threading import Thread
import queue
import json
import time
import geocoder
from datetime import datetime

acc_model   = YOLO(r"E:\My Projects\Accident\Final_run\Models\yolo_45k.pt")
plate_model = YOLO(r"E:\My Projects\Accident\Final_run\Models\license-plate-finetune-v1l.pt")

os.makedirs("cropped_plates", exist_ok=True)
os.makedirs("alert_frames",   exist_ok=True)
os.makedirs("alert_jsons",    exist_ok=True)

ACCIDENT_CLASSES = ["accident"]
PLATE_CLASSES    = ["license-plate"]

SKIP_FRAMES    = 4
INFER_SIZE     = 640
ALERT_COOLDOWN = 30

last_alert_time = 0


def get_location():
    try:
        g = geocoder.ip("me")
        if g.ok:
            lat, lng = g.latlng
            return {
                "latitude":  lat,
                "longitude": lng,
                "city":      g.city    or "Unknown",
                "state":     g.state   or "Unknown",
                "country":   g.country or "Unknown",
                "maps_link": f"https://www.google.com/maps?q={lat},{lng}"
            }
    except:
        pass
    return {
        "latitude":  None,
        "longitude": None,
        "city":      "Unknown",
        "state":     "Unknown",
        "country":   "Unknown",
        "maps_link": ""
    }


def save_alert(frame, frame_id, acc_boxes, plate_boxes):
    global last_alert_time
    now = time.time()
    if now - last_alert_time < ALERT_COOLDOWN:
        return None
    last_alert_time = now

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    frame_path = f"alert_frames/accident_{timestamp}_frame{frame_id}.jpg"
    cv2.imwrite(frame_path, frame)

    location = get_location()

    accidents = [
        {
            "label":      name,
            "confidence": round(conf, 3),
            "bbox":       {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        }
        for (x1, y1, x2, y2, name, conf) in acc_boxes
        if name.lower() in ACCIDENT_CLASSES
    ]

    plates = [
        {
            "label":      name,
            "confidence": round(conf, 3),
            "bbox":       {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        }
        for (x1, y1, x2, y2, name, conf) in plate_boxes
    ]

    alert = {
        "alert_id":   f"ACC_{timestamp}_F{frame_id}",
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "frame_id":   frame_id,
        "frame_path": frame_path,
        "location":   location,
        "accidents":  accidents,
        "plates":     plates,
        "status":     "ACCIDENT_DETECTED"
    }

    json_path = f"alert_jsons/alert_{timestamp}_frame{frame_id}.json"
    with open(json_path, "w") as f:
        json.dump(alert, f, indent=4)

    print(f"✅ Alert saved: {json_path}")
    return alert


def run_detector(video_path):
    cap    = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    sx = width  / INFER_SIZE
    sy = height / INFER_SIZE

    def scale_box(x1, y1, x2, y2):
        return int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)

    def expand_box(x1, y1, x2, y2, pad=40):
        return (
            max(0, x1 - pad),
            max(0, y1 - pad),
            min(width,  x2 + pad),
            min(height, y2 + pad),
        )

    frame_queue = queue.Queue(maxsize=8)

    def frame_reader():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                frame_queue.put(None)
                break
            frame_queue.put(frame)

    Thread(target=frame_reader, daemon=True).start()

    frame_id    = 0
    plate_count = 0
    last_acc_boxes   = []
    last_plate_boxes = []

    while True:
        frame = frame_queue.get()
        if frame is None:
            break

        frame_id += 1

        if frame_id % SKIP_FRAMES != 0:
            continue

        small = cv2.resize(frame, (INFER_SIZE, INFER_SIZE))

        # ── Accident Detection ───────────────────────────────
        acc_results = acc_model.predict(
            small, conf=0.1, iou=0.5,
            imgsz=INFER_SIZE, verbose=False, device="cpu"
        )

        last_acc_boxes    = []
        last_plate_boxes  = []
        accident_boxes    = []
        accident_detected = False

        for r in acc_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1, x2, y2 = scale_box(x1, y1, x2, y2)
                cls  = int(box.cls[0])
                conf = float(box.conf[0])
                name = acc_model.names[cls]
                last_acc_boxes.append((x1, y1, x2, y2, name, conf))
                if name.lower() in ACCIDENT_CLASSES:
                    accident_boxes.append((x1, y1, x2, y2))
                    accident_detected = True

        # ── Plate Detection ──────────────────────────────────
        if accident_detected:
            for (ax1, ay1, ax2, ay2) in accident_boxes:
                rx1, ry1, rx2, ry2 = expand_box(ax1, ay1, ax2, ay2)
                region = frame[ry1:ry2, rx1:rx2]

                if region.size == 0:
                    continue

                plate_results = plate_model.predict(
                    region, conf=0.3, iou=0.5,
                    imgsz=INFER_SIZE, verbose=False, device="cpu"
                )

                plate_found = False
                for pr in plate_results:
                    for pb in pr.boxes:
                        px1, py1, px2, py2 = map(int, pb.xyxy[0])
                        pname = plate_model.names[int(pb.cls[0])]
                        pconf = float(pb.conf[0])

                        if pname.lower() in PLATE_CLASSES:
                            abs_x1, abs_y1 = rx1 + px1, ry1 + py1
                            abs_x2, abs_y2 = rx1 + px2, ry1 + py2
                            last_plate_boxes.append((abs_x1, abs_y1, abs_x2, abs_y2, pname, pconf))

                            if frame_id % 30 == 0:
                                crop = frame[abs_y1:abs_y2, abs_x1:abs_x2]
                                if crop.size > 0:
                                    save_path = f"cropped_plates/plate_f{frame_id}_{plate_count}.jpg"
                                    cv2.imwrite(save_path, crop)
                                    print(f"✅ Plate saved: {save_path}")
                                    plate_count += 1
                                    plate_found = True

                if not plate_found and frame_id % 30 == 0:
                    save_path = f"cropped_plates/region_f{frame_id}_{plate_count}.jpg"
                    cv2.imwrite(save_path, region)
                    print(f"📸 Region saved: {save_path}")
                    plate_count += 1

            alert = save_alert(
                frame.copy(), frame_id,
                last_acc_boxes, last_plate_boxes
            )
            if alert:
                yield alert

    cap.release()
    print(f"\n✅ Done! {plate_count} plates saved.")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else r"E:\My Projects\Accident\Test Videos\12.mp4"
    for alert in run_detector(path):
        print(json.dumps(alert, indent=4))