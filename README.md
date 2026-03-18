# DashCam Accident Detection & Alert System

An AI-powered in-vehicle dashcam system that detects road accidents in real-time, captures evidence frames, retrieves the vehicle's GPS location, and instantly sends structured alerts to a control room dashboard.

---

## Problem Statement

Road accidents require immediate emergency response, but delays in reporting cost lives. Traditional dashcams only record footage — they do not detect accidents or notify anyone automatically. This system bridges that gap by using AI to detect accidents the moment they happen and alert a control room without any human intervention.

---

## How It Works

```
Dashcam Video Feed
       |
       v
  Accident Detection         <-- YOLOv8L Model
       |
       v
  License Plate Detection    <-- YOLOv8L Plate Model
       |
       v
  Fetch GPS Location         <-- IP Geolocation + Google Maps Link
       |
       v
  Generate Alert             <-- JSON File + Captured Frame (.jpg)
       |
       v
  Control Room Dashboard     <-- Streamlit Live UI
```

---

## Features

| Feature | Description |
|---|---|
| Accident Detection | YOLOv8L detects accidents in real-time from dashcam video |
| License Plate Detection | Identifies and crops license plates within the accident region |
| GPS Location | Fetches live location with a Google Maps link on accident trigger |
| Frame Capture | Saves the exact video frame at the moment of detection |
| JSON Alert | Structured alert with timestamp, GPS, plate info, and confidence scores |
| Control Room Dashboard | Live Streamlit UI showing frames, location, and full alert history |
| Alert Cooldown | Prevents duplicate alerts within a configurable time window |
| Threaded Processing | Non-blocking video pipeline for smooth real-time performance |

---

## Project Structure

```
dashcam-accident-detection/
|
|-- detector.py          # Core AI detection engine
|-- app.py               # Control room Streamlit dashboard
|-- requirements.txt     # Python dependencies
|-- README.md            # Project documentation
|
|-- alert_frames/        # Captured accident frames (.jpg)
|-- alert_jsons/         # Structured alert JSON files
|-- cropped_plates/      # Cropped license plate images
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Object Detection | YOLOv8L (Ultralytics) |
| Video Processing | OpenCV |
| Dashboard UI | Streamlit |
| Location Fetch | Geocoder (IP-based) |
| Alert Format | JSON |
| Language | Python 3.10+ |
| GPU Support | NVIDIA CUDA 12.8 |

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/yourusername/dashcam-accident-detection.git
cd dashcam-accident-detection
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. GPU support (CUDA 12.8)**

```bash
pip install torch==2.10.0+cu128 torchvision==0.20.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
```

---

## Configuration

Update the model paths in `detector.py`:

```python
acc_model   = YOLO(r"path/to/accident_model.pt")
plate_model = YOLO(r"path/to/plate_model.pt")
```

**Tunable parameters:**

| Parameter | Default | Description |
|---|---|---|
| SKIP_FRAMES | 4 | Process every Nth frame |
| INFER_SIZE | 640 | Inference image resolution |
| ALERT_COOLDOWN | 30 | Seconds between consecutive alerts |
| ACCIDENT_CLASSES | ["accident"] | Class label for accident detection |
| PLATE_CLASSES | ["license-plate"] | Class label for plate detection |

---

## Running the Dashboard

```bash
streamlit run app.py
```

Open browser at `http://localhost:8501`

1. Upload dashcam footage (.mp4, .avi, .mov, .mkv)
2. Preview the uploaded video
3. Click Start Detection
4. Alerts populate live on the dashboard as accidents are detected

---

## Alert JSON Structure

```json
{
    "alert_id":   "ACC_20240315_142301_F120",
    "timestamp":  "2024-03-15 14:23:01",
    "frame_id":   120,
    "frame_path": "alert_frames/accident_20240315_142301_frame120.jpg",
    "location": {
        "latitude":  9.9312,
        "longitude": 76.2673,
        "city":      "Kochi",
        "state":     "Kerala",
        "country":   "India",
        "maps_link": "https://www.google.com/maps?q=9.9312,76.2673"
    },
    "accidents": [
        {
            "label":      "accident",
            "confidence": 0.874,
            "bbox":       { "x1": 200, "y1": 150, "x2": 600, "y2": 450 }
        }
    ],
    "plates": [
        {
            "label":      "license-plate",
            "confidence": 0.791,
            "bbox":       { "x1": 412, "y1": 310, "x2": 498, "y2": 340 }
        }
    ],
    "status": "ACCIDENT_DETECTED"
}
```

---

## Output Files

| Folder | File Type | Description |
|---|---|---|
| alert_frames/ | .jpg | Frame captured at moment of accident |
| alert_jsons/ | .json | Full structured incident report |
| cropped_plates/ | .jpg | Cropped license plate image |

---

## AI Models

| Model | Task | Architecture |
|---|---|---|
| Accident Detection | Detects road accidents in video frames | YOLOv8L |
| Plate Detection | Locates license plates within the accident region | YOLOv8L |

**Training the accident model:**

```bash
yolo detect train data=data.yaml model=yolov8l.pt epochs=100 imgsz=640 batch=64 device=0
```

---

## Troubleshooting

| Issue | Solution |
|---|---|
| Model not loading | Verify .pt file paths in detector.py |
| No accidents detected | Lower conf to 0.05, check class name matches model |
| Slow processing | Increase SKIP_FRAMES, switch to GPU with device=0 |
| Location shows Unknown | Check internet connection, geocoder may be rate-limited |

---

## Future Enhancements

- Live RTSP dashcam stream support
- SMS or email notifications to control room
- OCR on license plate crops for text extraction
- Multi-vehicle GPS tracking on a map
- Accident frequency analytics and reporting
- Cloud upload of alerts to AWS S3 or Firebase

---

## License

MIT License. Free to use, modify, and distribute.

---



Built using [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) and [Streamlit](https://streamlit.io).
