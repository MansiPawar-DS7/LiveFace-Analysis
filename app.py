import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
from comb_model import process_frame

# PAGE CONFIG 
st.set_page_config(
    page_title="AI Face Analytics",
    page_icon="🎀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# HEADER 
st.markdown("<h1 style='text-align:center;'>🎀 AI Face Analytics 🎀</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Turn on your camera to reveal your age, mood, and more!</p>", unsafe_allow_html=True)

# LAYOUT
col1, col2 = st.columns([3,1])

# RIGHT PANEL 
with col2:

    st.markdown("""
    <div style="background:#FFF9C4;padding:20px;border-radius:20px;">
    <b>💡 Tips for Best Results:</b>
    <ul>
        <li>Allow camera permissions</li>
        <li>Ensure good lighting</li>
        <li>Stay still</li>
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

# 🎥 WebRTC Video Processor
class VideoProcessor(VideoTransformerBase):

    def __init__(self):
        self.latest_message = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        processed_frame, emotion, message = process_frame(img)

        # Store in session_state (IMPORTANT)
        if message:
            self.latest_message = message
        return processed_frame

# LEFT PANEL (CAMERA)
with col1:

    st.markdown("### 🎥 Live Camera")
    st.write("")

    camera_placeholder = st.empty()
    message_placeholder = st.empty()

    RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]}
        ]
    }
    )
    
    webrtc_ctx = webrtc_streamer(
        key="live-camera",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration=RTC_CONFIGURATION
    )

    # 🔥 LIVE MESSAGE LOOP (THIS FIXES EVERYTHING)
    if webrtc_ctx.state.playing:
        while True:
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
                        ">
                            💬 {message}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                message_placeholder.write("")

            import time
            time.sleep(0.5)

    # Camera OFF UI
    if not webrtc_ctx.state.playing:
        camera_placeholder.markdown("""
        <div style="
            background:#F0F4F8;
            height:480px;
            border-radius:20px;
            display:flex;
            align-items:center;
            justify-content:center;
            flex-direction:column;
            border:3px dashed #CFD8DC;
        ">
            <h2>📷 Camera is Off</h2>
            <p>Click START below</p>
        </div>
        """, unsafe_allow_html=True)

# FOOTER 
st.markdown("---")
st.markdown("<p style='text-align:center;'>👩‍💻 TY Project | 2026 | M.P </p>", unsafe_allow_html=True)
