print("THIS FILE IS RUNNING:", __file__)

import cv2
import numpy as np
import tensorflow as tf
import random
from collections import deque
from tensorflow.keras.applications.resnet50 import preprocess_input

import os
import gdown

IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.25

BASE_DIR = os.path.dirname(__file__)

AGE_MODEL_PATH = os.path.join(BASE_DIR, "models","training_age_model.h5")
GENDER_MODEL_PATH = os.path.join(BASE_DIR, "models","mobilenetv2_utkface_gender.h5")
EMOTION_MODEL_PATH = os.path.join(BASE_DIR, "models","best_emotion_model.h5")

# ensure folder exists (VERY IMPORTANT FOR STREAMLIT CLOUD)
os.makedirs(os.path.dirname(EMOTION_MODEL_PATH), exist_ok=True)

if not os.path.exists(EMOTION_MODEL_PATH):
    url = "https://drive.google.com/uc?id=1sMqDb8zopHNa2m_Q3xNM84lQmmFZR1de"
    gdown.download(url, EMOTION_MODEL_PATH, quiet=False)


# LABELS
emotion_labels = [
    "Anger", "Disgust", "Fear",
    "Happy", "Neutral", "Sad", "Surprise"
]

gender_labels = ["female", "male"]

age_groups = [
    "0-10", "11-20", "21-30", "31-40",
    "41-50", "51-60", "61-70",
    "71-80", "81-90", "90+"
]

# LOAD MODELS
print("Loading models...")

age_model = tf.keras.models.load_model(AGE_MODEL_PATH)
gender_model = tf.keras.models.load_model(GENDER_MODEL_PATH)
emotion_model = tf.keras.models.load_model(EMOTION_MODEL_PATH)

print("Models loaded successfully!")

# EMOTION SMOOTHING BUFFER
emotion_buffer = deque(maxlen=10)     #the emotion will not change constantly like every second

# MOTIVATIONAL MESSAGES
def get_motivational_message(emotion):
    messages = {
        "Anger": ["Pause. Breathe. Your feelings are valid, but you're stronger than this moment."],
        "Fear": ["It's okay to be scared. Courage starts exactly where fear exists."],
        "Sad": ["This feeling will pass. Be kind to yourself—you’re doing your best."],
        "Happy": ["Hold onto this joy. Moments like this are proof that life is beautiful."],
        "Neutral": ["Calm is powerful. Take this moment to reset and move forward."],
        "Surprise": ["Unexpected moments can lead to new opportunities. Stay open."],
        "Disgust": ["Step back, breathe, and protect your peace. You deserve comfort."],
        "Uncertain": ["It's okay not to have all the answers. Clarity will come, one step at a time."]
    }

    return random.choice(messages.get(emotion, ["Stay positive."]))

# PREPROCESS FUNCTIONS

# Age + Gender (MobileNetV2)
def preprocess_basic(face):         #Resizing, Normalizing

    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    face = face / 255.0
    face = np.expand_dims(face, axis=0)

    return face


# Emotion (ResNet50)
def preprocess_emotion(face):        #Pixel scaling, Normalizing

    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    face = np.expand_dims(face, axis=0)
    face = preprocess_input(face.astype("float32"))

    return face

# FACE DETECTOR
face_cascade = cv2.CascadeClassifier(                                    #Uses OpenCV’s Haar Cascade for detecting faces in each video frame
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml" 
)

# MAIN PROCESS FUNCTION
def process_frame(frame):

    emotion = None
    message = None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)       #covert image into greyscale for face detection

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  #detect face

    for (x, y, w, h) in faces:

        face = frame[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)      #detect the face and convert into RGB

        face_basic = preprocess_basic(face_rgb)
        face_emotion = preprocess_emotion(face_rgb)

        #EMOTION
        emotion_pred = emotion_model.predict(face_emotion, verbose=0)[0]

        emotion_pred = emotion_pred / np.sum(emotion_pred)

        emo_idx = np.argmax(emotion_pred)                    #np.argmax: index of the highest probability class.
        emo_conf = emotion_pred[emo_idx]                     #it showa the confidence score of the class like happy -> 0.75

        if emo_conf < CONFIDENCE_THRESHOLD:
            emotion = "Uncertain"
        else:
            emotion = emotion_labels[emo_idx]

        # Emotion smoothing
        emotion_buffer.append(emotion)
        emotion = max(set(emotion_buffer), key=emotion_buffer.count)

        # GENDER
        gender_pred = gender_model.predict(face_basic, verbose=0)    #similar logic to gender and age

        if gender_pred.shape[1] == 1:
            gender = "female" if gender_pred[0][0] < 0.5 else "male"
        else:
            gender = gender_labels[np.argmax(gender_pred)]

        # AGE
        age_pred = age_model.predict(face_basic, verbose=0)

        if age_pred.shape[1] > 1:
            age = age_groups[np.argmax(age_pred)]
        else:
            age = int(age_pred[0][0] * 100)

        # MESSAGE
        message = get_motivational_message(emotion)     #calling message function that gives message based on emotion

        # DRAW  --- it will draw a box around a face along with the age, gender and emotion
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        cv2.putText(frame, f"Age: {age}", (x, y-60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        cv2.putText(frame, f"Gender: {gender}", (x, y-35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        cv2.putText(frame, f"Emotion: {emotion} ({emo_conf:.2f})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        break
        # If no face detected

    if message is None:
        emotion = "Uncertain"
        message = get_motivational_message(emotion)

    return frame, emotion, message
