import streamlit as st
import cv2
import time
from comb_model import process_frame 

# PAGE CONFIG 
st.set_page_config(                           #this is about the page
    page_title="AI Face Analytics",
    page_icon="🎀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# SESSION STATE 
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False

if 'cap' not in st.session_state:
    st.session_state.cap = None

# HEADER 
st.markdown("<h1 style='text-align:center;'>🎀 AI Face Analytics 🎀</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Turn on your camera to reveal your age, mood, and more!</p>", unsafe_allow_html=True)

# LAYOUT
col1, col2 = st.columns([3,1])      #col 1(left) > main camera display | col 2(right) > info panel, buttons, tips
 
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

    # Start / Stop camera buttons
    if not st.session_state.camera_active:
        if st.button("Start Camera 📸"):
            st.session_state.camera_active = True
    else:
        if st.button("Stop Camera ⏹️"):
            st.session_state.camera_active = False
            if st.session_state.cap:
                st.session_state.cap.release()
                st.session_state.cap = None

    st.write("")

    st.markdown("""
    <div style="background:white;padding:20px;border-radius:20px;border:2px solid #E2F0CB;">
    <b>✨ Model Features</b><br><br>
    👶 Age Prediction<br>
    🚻 Gender Classification<br>
    😊 Emotion Recognition
    </div>
    """, unsafe_allow_html=True)

# LEFT PANEL (CAMERA)
with col1:

    video_placeholder = st.empty()        #camera place
    message_placeholder = st.empty()      #message place

    if st.session_state.camera_active:
        
        if st.session_state.cap is None:
            st.session_state.cap = cv2.VideoCapture(0)

        cap = st.session_state.cap

        # Stream frames
        while st.session_state.camera_active:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to capture frame")
                break

            # Process frame
            processed_frame, emotion, message = process_frame(frame)

            # Display
            video_placeholder.image(
                processed_frame,
                channels="BGR",
                use_container_width=True
            )

            if message:
                message_placeholder.markdown(
                    f"<div style='background:#E0F2F1;padding:15px;border-radius:15px;text-align:center;'>💬 {message}</div>",
                    unsafe_allow_html=True
                )

            time.sleep(0.03)  # small delay to reduce CPU usage

        # Release camera after stopping
        cap.release()
        st.session_state.cap = None

    else:
        video_placeholder.markdown("""
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
            <p>Click Start Camera</p>
        </div>
        """, unsafe_allow_html=True)

# FOOTER 
st.markdown("---")
st.markdown("<p style='text-align:center;'>👩‍💻 TY Project | 2026 | M.P </p>", unsafe_allow_html=True)