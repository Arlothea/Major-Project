import os

# ------------------ ENVIRONMENTAL VARIABLES ------------------
# These environmental variables help reduce tensorflow console spam and forces CPU usage.
# These are needed due to the current hardware limitations on the device.
# TF_CPP_MIN_LOG_LEVEL=3 suppresses all logs except errors.
# CUDA_VISIBLE_DEVICES=-1 forces TensorFlow to use the CPU instead of any available GPUs.
# TF_ENABLE_ONEDNN_OPTS=0 disables some CPU optimizations which can cause issues.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import time
import sys
import cv2
import numpy as np
import requests
import socket

# Import the EmotionRecognizerModel from the Runtime module.
from Runtime.EmotionRecognizer import EmotionRecognizerModel

# ------------------ FACE DETECTION MODEL ------------------
# Haar Cascade is a classical detector thast is runs in realtime on CPU.
# It detects where faces are in the frame, allowing us to focus our emotion recognition on those areas.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ------------------ GLOBAL VARIABLES ------------------
# These are kept globally so that the initialise only once and reused each frame
recognizer = None
cap = None

# ------------------- ESCALATION VARIABLES ------------------
# This section of varioables is used to track the state of the system and determine when to trigger an escalation.
# Calm -> Angry transition triggers and escalation alert.
# Alert stays visible for 2 seconds after the transition is detected.
# Angry frame count is used to ensure that the system is accurate when detecting escalation.
alert_active = False
alert_start_time = 0
ALERT_DURATION = 2
angry_frame_count = 0

# ---------------- FACE TRACKING VARIABLES ------------------
# Face persistence allows the system to keep tracking a face for a short period of time after the face disappears.
# Smoothing helps stabilise the detected face coordinates to avoid jitters.
FACE_PERSISTENCE = 1.5
SMOOTHING = 0.8

# ----------------- EMOTION HISTORY VARIABLES ------------------
# last state remembers the previous emotional state to detect transitions.
# transition_detected becomes tue when a calm -> angry transition is detected.
# last_faces and last_face_time stores last detected faces and the time seen.
# Emotion history is used to smooth out the emotion predictions over a short window of frames(5) to improve accuracy.
last_state = "calm"
transition_detected = False
last_faces = []
last_face_time = 0.0
emotion_history = []
EMOTION_WINDOW = 5

def get_ipv4():
    # This function gets the local IPv4 address of the device by opening a UDP socket.
    # This allows the system to detect its own IP address without using hardcoded IP addresses.
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip

SERVER_IP = get_ipv4()
API_SERVER = f"http://{SERVER_IP}:8000"

def send_escalation(name="Unknown", level=2, camera="Room1"):
    # This function sends an escalation alert to the API server with the given name, level, and camera information.
    try:
        requests.post(
            f"{API_SERVER}/escalation",
            json={"name": name, "level": level, "camera": camera},
            timeout=0.3
        )
    except:
        # If escalation fails, this ignores the error and continues running.
        pass

# Loads the emotion recognition model once and keeps it in memory for future predictions.
def _ensure_recognizer(model_path = None):
    global recognizer

    # If the recognizer is already initialized, do nothing
    if recognizer is not None:
        return

    # Designates the model path if not provided
    if model_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "Models", "emotion_model.h5")

        recognizer = EmotionRecognizerModel(model_path)

# Opens a camera stream and keeps it open.
def _ensure_camera(device_index=0):
    global cap

    # If the camera is already opened, do nothing
    if cap is not None and cap.isOpened():
        return True

    # Try different backends based on the operating system, in this instance 'windows'.
    backends = []
    if sys.platform.startswith("win"):
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_ANY]

    # Attempt to open the camera with each backend until one succeeds
    for backend in backends:
            if backend ==  cv2.CAP_ANY:
                cap = cv2.VideoCapture(device_index)
            else:
                cap = cv2.VideoCapture(device_index, backend)

            if cap is not None and cap.isOpened():
                return True

            # Release the capture if it failed and then try the next backend.
            if cap is not None:
                cap.release()
            cap = None

    # As a last resort, try opening the default camera (index 0)
    if device_index != 0:
            cap = cv2.VideoCapture(0)
            if cap is not None and cap.isOpened():
                return True

            if cap is not None:
                cap.release()
            cap = None
    return False

def get_frame():
    # get_frame() Captures a frame from the camera, detects faces, and recognize emotions.
    # It then predicts the emotion for each face and shows the result on the frame.
    global cap, face_cascade, recognizer
    global last_faces, last_face_time
    global last_state, transition_detected
    global alert_active, alert_start_time

    try:
    # --------------- CAMERA AND MODEL INITIALIZATION ---------------
    # Ensure the camera is initialized
        if not _ensure_camera():
            # If the camera cannot be opened, return a blank frame and empty results.
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            return blank, []

        # Ensure the emotion recognizer model is loaded
        if recognizer is None:
            _ensure_recognizer()

        # --------------- FRAME CAPTURE ---------------
        # Capture a frame from the camera
        ret, frame = cap.read()

        # If frame capture failed, return a blank frame and empty results
        if not ret or frame is None:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            return blank, []

        results = []

        # --------------- FACE DETECTION ---------------
        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        # Resize the grayscale image for faster face detection.
        scale = 0.75
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation = cv2.INTER_LINEAR)

        # Detect faces in a rectangular area of the frame using the Haar Cascade detector.
        faces = face_cascade.detectMultiScale(small, 1.08, 8, 0, (80, 80))

        # Scale the face coordinates back to the original frame size
        faces = [(int(x/scale), int(y/scale), int(w/scale), int(h/scale)) for (x, y, w, h) in faces]

        # --------------- FACE FILTERING ---------------
        # Filter faces based on aspect ratio and size constraints
        # This reduces false positives and focuses on faces that are more likely to be valid for emotion recognition.
        H,W = frame.shape[:2]
        filtered = []
        for (x, y, w, h) in faces:
            ar = w / float(h)

            # Aspect ratio between 0.75 and 1.33
            if ar < 0.75 or ar > 1.33:
                continue

            # Size constraints: between 12% and 80% of frame dimensions
            # ignores faces that are too small or too large.
            if w < 0.12 * W or h < 0.12 * H:
                continue
            if w > 0.80 * W or h > 0.80 * H:
                continue

            filtered.append((x, y, w, h))
        faces = filtered

        # --------------- FACE TRACKING AND SMOOTHING ---------------
        now = time.time()

        # Smooth the facial recognition area for more accurate detection.
        if len(faces) > 0:
            if last_faces:
                smoothed = []
                for (x, y, w, h), (lx, ly, lw, lh) in zip(faces, last_faces):
                    sx = int(SMOOTHING * lx + (1 - SMOOTHING) * x)
                    sy = int(SMOOTHING * ly + (1 - SMOOTHING) * y)
                    sw = int(SMOOTHING * lw + (1 - SMOOTHING) * w)
                    sh = int(SMOOTHING * lh + (1 - SMOOTHING) * h)
                    smoothed.append((sx, sy, sw, sh))
                faces = smoothed

            last_faces = faces
            last_face_time = now

        else:
            # If detection fails , keep using the last detected faces for a short period of time.
            if last_faces and (now - last_face_time < FACE_PERSISTENCE):
                faces = last_faces
            else:
                faces = []    

        # --------------- EMOTION RECOGNITION AND ANNOTATION ---------------
        # For each detected face, predict the emotion and annotate the frame
        for (x, y, w, h) in faces:
            face_bgr = frame[y : y + h, x : x + w]           
            if face_bgr.size == 0:
                continue

            # Preprocess the face region for facial emotion recognition input (48x48).
            face_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
            face_gray = cv2.resize(face_gray, (48, 48), interpolation = cv2.INTER_AREA)
            
            # Model expects 3 channels, so convert the grayscale face back to BGR format.
            face_input = cv2.cvtColor(face_gray, cv2.COLOR_GRAY2BGR)

            try:
                # predict returns label and confidence in percentage.
                pred = recognizer.predict(face_input)
                if pred is None or len(pred) < 2:
                    continue

                raw_emotion, confidence = pred
                confidence = float(confidence)

                # ----------------- EMOTION SMOOTHING -----------------
                emotion_history.append(raw_emotion)
                if len(emotion_history) > EMOTION_WINDOW:
                    emotion_history.pop(0)
                
                # The most common emotion in the recent history is chosen as the display emotion to smooth out predictions over time.
                display_emotion = max(set(emotion_history), key=emotion_history.count)

                # --------------- ESCALATION DETECTION ---------------
                # This requires a certain amount of consecutive frames before swicthing to angry state.
                if raw_emotion.lower() == "angry":
                    angry_frame_count += 1
                else:
                    angry_frame_count = 0

                current_state = "angry" if angry_frame_count >= 5 else "calm"

                # Detects calm -> angry transition to trigger escalation alert.
                transition_detected = False
                if last_state == "calm" and current_state == "angry":
                    transition_detected = True
                last_state = current_state

                results.append((display_emotion, confidence))

                # --------------- FRAME ANNOTATION ---------------
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)   
                cv2.putText(frame, display_emotion, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

                # --------------- ESCALATION ALERT DISPLAY ---------------
                current_time = time.time()
                if transition_detected and not alert_active:
                    alert_active = True
                    alert_start_time = current_time
                    # This is where the system sends an escalation alert to the API server when a calm -> angry transition is detected.
                    send_escalation(name="Child Escalated", level=2, camera="Acorn")

                # Show alert text for a short duration after escalation detection.
                if alert_active:
                    if current_time - alert_start_time < ALERT_DURATION:
                        cv2.putText(frame, "ESCALATION DETECTED",
                        (x, y - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0,0,255), 2)
                    else:
                        alert_active = False

            except Exception:
                # If the prediction fails, skip the the face and keep running.
                continue

        return frame, results
    except Exception as e:
        # Global exception handling to catch any unexpected errors during the frame processing.
        # Prevents the application from crashing.
        print("Error in get_frame:", e)
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        return blank, []

# Release the camera resource.
def release():
    global cap
    if cap is not None:
        cap.release()
        cap = None