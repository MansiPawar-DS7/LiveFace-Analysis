import streamlit as st
import cv2
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

    st.write("")

    st.markdown("""
    <div style="background:#FFEBEE;padding:20px;border-radius:20px;border:2px solid #EF9A9A;">
    <b>⚠️ Important Note</b><br><br>
    If the camera gets stuck or your face is not detected:<br><br>
    👉 Stop the camera and restart it<br>
    </div>
    """, unsafe_allow_html=True)


# LEFT PANEL
with col1:

    st.markdown("### 🎥 Live Camera")

    frame_placeholder = st.empty()
    message_placeholder = st.empty()

    # SESSION STATE INIT
    if "run" not in st.session_state:
        st.session_state.run = False

    # TOGGLE BUTTON (START ↔ STOP)
    if st.session_state.run:
        if st.button("⛔ Stop Camera"):
            st.session_state.run = False
    else:
        if st.button("▶ Start Camera"):
            st.session_state.run = True

    cap = None

    # CAMERA ON
    if st.session_state.run:

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("❌ Cannot access camera")
        else:
            while st.session_state.run:

                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read frame")
                    break

                # PROCESS FRAME (YOUR MODEL)
                processed_frame, emotion, message = process_frame(frame)

                # FIXED SIZE CAMERA FRAME
                processed_frame = cv2.resize(processed_frame, (720, 480))

                # SHOW VIDEO
                frame_placeholder.image(processed_frame, channels="BGR")

                # SHOW MESSAGE
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

        cap.release()

    # CAMERA OFF UI
    else:
        frame_placeholder.markdown("""
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
            <p>Click Start to begin detection</p>
        </div>
        """, unsafe_allow_html=True)

# FOOTER
st.markdown("---")
st.markdown("<p style='text-align:center;'>👩‍💻 TY Project | 2026 | M.P </p>", unsafe_allow_html=True)
