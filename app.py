import streamlit as st
import json
import os
import tempfile
import cv2
from detector import run_detector

st.set_page_config(
    page_title="Accident Detection System",
    page_icon="🚨",
    layout="wide"
)

st.markdown("""
    <h1 style='text-align:center; color:red;'>🚨 Accident Detection System</h1>
    <p style='text-align:center; color:gray;'>AI-powered real-time accident & license plate detection</p>
    <hr style='border: 1px solid red;'>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/emoji/96/police-car-light.png", width=80)
    st.title("⚙️ Settings")
    st.markdown("---")
    st.markdown("**Model Info**")
    st.info("🧠 YOLOv8L — Accident Model\n\n🧠 YOLOv8L — Plate Model")
    st.markdown("---")
    st.markdown("**Detection Settings**")
    conf_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.25, 0.05)
    cooldown       = st.slider("Alert Cooldown (sec)",    5,  120,   30,    5)
    st.markdown("---")
    st.markdown("**Output Folders**")
    st.code("alert_frames/\nalert_jsons/\ncropped_plates/")

# ── Video Upload ─────────────────────────────────────────────
st.subheader("📂 Upload Video")
uploaded_file = st.file_uploader(
    "Upload a dashcam or CCTV video",
    type=["mp4", "avi", "mov", "mkv"]
)

if not uploaded_file:
    st.warning("👆 Please upload a video file to begin.")
    st.stop()

with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
    tmp.write(uploaded_file.read())
    video_path = tmp.name

st.success(f"✅ Uploaded: `{uploaded_file.name}` — Ready to process!")

with st.expander("▶️ Preview Uploaded Video"):
    st.video(uploaded_file)

st.markdown("---")

# ── Start Button ─────────────────────────────────────────────
start = st.button("🚀 Start Detection", type="primary", width="stretch")

if start:

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("📷 Accident Frame")
        frame_placeholder = st.empty()
        frame_placeholder.info("⏳ Waiting for first accident...")

    with col2:
        st.subheader("📋 Alert Details")
        alert_placeholder = st.empty()
        alert_placeholder.info("⏳ No alerts yet...")

    st.markdown("---")
    st.subheader("📜 Alert History")
    log_placeholder   = st.empty()
    count_placeholder = st.empty()

    alert_history = []
    alert_count   = 0

    st.toast("🟢 Detection started!", icon="🚨")

    for alert in run_detector(video_path):

        alert_count += 1
        loc       = alert.get("location",  {})
        plts      = alert.get("plates",    [])
        acc       = alert.get("accidents", [])
        maps_link = loc.get("maps_link",   "")

        # ── Left: Frame ──────────────────────────────────────
        with col1:
            frame_path = alert.get("frame_path", "")
            if os.path.exists(frame_path):
                frame = cv2.imread(frame_path)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(
                    frame,
                    caption=f"🚨 Alert #{alert_count}  |  Frame: {alert['frame_id']}  |  {alert['timestamp']}",
                    width="stretch"
                )

        # ── Right: Details ───────────────────────────────────
        with col2:
            alert_placeholder.empty()
            with alert_placeholder.container():

                st.error(f"🚨 ACCIDENT #{alert_count} DETECTED!")
                st.markdown(f"🕐 **Time:** `{alert['timestamp']}`")
                st.markdown(f"🎞 **Frame ID:** `{alert['frame_id']}`")
                st.markdown(f"🆔 **Alert ID:** `{alert['alert_id']}`")
                st.markdown("---")
                st.markdown("📍 **Location**")

                loc_col1, loc_col2 = st.columns(2)
                with loc_col1:
                    st.metric("🏙 City",    loc.get("city",    "N/A"))
                    st.metric("🗺 State",   loc.get("state",   "N/A"))
                with loc_col2:
                    st.metric("🌍 Country", loc.get("country", "N/A"))
                    st.metric("📌 GPS",     f"{loc.get('latitude')}, {loc.get('longitude')}")

                if maps_link:
                    st.link_button("🗺 Open in Google Maps", maps_link, width="stretch")

                st.markdown("---")

                if acc:
                    st.markdown("💥 **Accident Detections**")
                    for a in acc:
                        st.progress(
                            a["confidence"],
                            text=f"`{a['label']}` — {a['confidence']*100:.1f}%"
                        )

                st.markdown("🚗 **License Plates**")
                if plts:
                    for p in plts:
                        st.progress(
                            p["confidence"],
                            text=f"`{p['label']}` — {p['confidence']*100:.1f}%"
                        )
                else:
                    st.warning("No plate detected in this frame.")

                with st.expander("🔍 View Raw JSON"):
                    st.json(alert)

        # ── History Table ────────────────────────────────────
        alert_history.append({
            "No":        alert_count,
            "Alert ID":  alert["alert_id"],
            "Time":      alert["timestamp"],
            "City":      loc.get("city",    "N/A"),
            "State":     loc.get("state",   "N/A"),
            "GPS":       f"{loc.get('latitude')}, {loc.get('longitude')}",
            "Plates":    len(plts),
            "Maps Link": maps_link
        })

        count_placeholder.success(f"🚨 Total Alerts Detected: **{alert_count}**")
        log_placeholder.dataframe(alert_history, width="stretch")

    st.balloons()
    st.success(f"✅ Detection complete! Total accidents found: **{alert_count}**")
    os.unlink(video_path)