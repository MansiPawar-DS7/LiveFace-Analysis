import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import av
from comb_model import process_frame
import time

# PAGE CONFIG
st.set_page_config(
    page_title="AI Face Analytics",
    page_icon="🎀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# HEADER
st.markdown("<h1 style='text-align:center;'>🎀 AI Face Analytics 🎀</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;'>Turn on your camera to reveal your age, mood, and more!</p>",
    unsafe_allow_html=True
)

# LAYOUT
col1, col2 = st.columns([3, 1])

# RIGHT PANEL
with col2:

    st.markdown("""
    <div style="background:#FFF9C4;padding:20px;border-radius:20px;">
    <b>💡 Tips for Best Results:</b>
    <ul>
        <li>Allow camera permissions</li>
        <li>Ensure good lighting</li>
        <li>Stay still</li>
        <li>Keep face centered</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    st.markdown("""
    <div style="background:white;padding:20px;border-radius:20px;border:2px solid #E2F0CB;">
    <b>✨ Model Features</b><br><br>
    👶 Age Prediction<br>
    🚻 Gender Classification<br>
    😊 Emotion Recognition
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    st.markdown("""
    <div style="background:#FFEBEE;padding:20px;border-radius:20px;border:2px solid #EF9A9A;">
    <b>⚠️ Important Note</b><br><br>
    If camera freezes or face is not detected:<br><br>
    👉 Stop camera and restart it
    </div>
    """, unsafe_allow_html=True)

# RTC CONFIG (important for deployment)
RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
        ]
    }
)

# VIDEO PROCESSOR
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.latest_message = "Waiting for detection..."

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        try:
            # Resize for better speed
            img = cv2.resize(img, (640, 480))

            processed_frame, emotion, message = process_frame(img)

            if message:
                self.latest_message = message

            return processed_frame

        except Exception as e:
            self.latest_message = "Processing error"
            return img

# LEFT PANEL
with col1:

    st.markdown("### 🎥 Live Camera")

    message_placeholder = st.empty()

    webrtc_ctx = webrtc_streamer(
        key="live-camera",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )

    # LIVE MESSAGE DISPLAY
    if webrtc_ctx.state.playing:

        while webrtc_ctx.state.playing:

            if webrtc_ctx.video_processor:
                message = webrtc_ctx.video_processor.latest_message

                if message:
                    message_placeholder.markdown(
                        f"""
                        <div style="
                            background:#E0F2F1;
                            padding:15px;
                            border-radius:15px;
                            text-align:center;
                            font-size:18px;
                            margin-top:10px;
                        ">
                            💬 {message}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            time.sleep(1)

    else:
        st.markdown("""
        <div style="
            background:#F0F4F8;
            height:480px;
            border-radius:20px;
            display:flex;
            align-items:center;
            justify-content:center;
            flex-direction:column;
            border:3px dashed #CFD8DC;
            margin-top:10px;
        ">
            <h2>📷 Camera is Off</h2>
            <p>Click START above to begin</p>
        </div>
        """, unsafe_allow_html=True)

# FOOTER
st.markdown("---")
st.markdown(
    "<p style='text-align:center;'>👩‍💻 TY Project | 2026 | M.P </p>",
    unsafe_allow_html=True
)
