import os
# Suppress TensorFlow logging and disable GPU usage
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import time
import sys
import cv2
import numpy as np

# Import the EmotionRecognizerModel from the Runtime module
from Runtime.EmotionRecognizer import EmotionRecognizerModel

# Load the pretrained XML model for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Global variables for the emotion recognizer and video capture
recognizer = None
cap = None

# GLOBALS
angry_frame_count = 0   

FACE_PERSISTENCE = 1.5
SMOOTHING = 0.8

last_state = "calm"
transition_detected = False

emotion_history = []
EMOTION_WINDOW = 5

last_faces = []
last_face_time = 0.0

# Ensure the emotion recognizer model is loaded
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

# Ensure the camera is initialized and opened
def _ensure_camera(device_index=0):

    # Use the global video capture object
    global cap

    # If the camera is already opened, do nothing
    if cap is not None and cap.isOpened():
        return True

    # Try different backends based on the operating system, in this instance 'windows'
    backends = []
    if sys.platform.startswith("win"):
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_ANY]

    # Attempt to open the camera with each backend until one succeeds
    for backend in backends:
            
            # CAP_ANY means OpenCV chooses automatically
            if backend ==  cv2.CAP_ANY:
                cap = cv2.VideoCapture(device_index)
            else:
                cap = cv2.VideoCapture(device_index, backend)

            if cap is not None and cap.isOpened():
                return True

            # Release the capture if it failed and then try the next backend
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

    # If all attempts fail, return False
    return False

def get_frame():
    # get_frame() Captures a frame from the camera, detects faces, and recognize emotions
    # It then predicts the emotion for each face and shows the result on the frame.
    global cap, face_cascade, recognizer
    global last_faces, last_face_time
    global last_state, transition_detected

    try:
    # Ensure the camera is initialized
        if not _ensure_camera():
            # If the camera cannot be opened, return a blank frame and empty results
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            return blank, []

        # Ensure the emotion recognizer model is loaded
        if recognizer is None:
            _ensure_recognizer()

        # Capture a frame from the camera
        ret, frame = cap.read()

        # If frame capture failed, return a blank frame and empty results
        if not ret or frame is None:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            return blank, []

        results = []

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        # Resize the grayscale image for faster face detection
        scale = 0.75
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation = cv2.INTER_LINEAR)

        # Detect faces in the grayscale frame
        # 1.08 is the scale factor 
        # 8 is the minNeighbors parameter
        faces = face_cascade.detectMultiScale(small, 1.08, 8, 0, (80, 80))

        # Scale the face coordinates back to the original frame size
        faces = [(int(x/scale), int(y/scale), int(w/scale), int(h/scale)) for (x, y, w, h) in faces]

        # Filter faces based on aspect ratio and size constraints
        H,W = frame.shape[:2]
        filtered = []
        for (x, y, w, h) in faces:
            ar = w / float(h)

            # Aspect ratio between 0.75 and 1.33
            if ar < 0.75 or ar > 1.33:
                continue

            # Size constraints: between 12% and 80% of frame dimensions
            if w < 0.12 * W or h < 0.12 * H:
                continue
            if w > 0.80 * W or h > 0.80 * H:
                continue

            # If all conditions are met, keep the face
            filtered.append((x, y, w, h))

        faces = filtered

        # Use last known faces if no faces are detected in the current frame
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
            if last_faces and (now - last_face_time < FACE_PERSISTENCE):
                faces = last_faces
            else:
                faces = []    

        # For each detected face, predict the emotion and annotate the frame
        for (x, y, w, h) in faces:
            face_bgr = frame[y : y + h, x : x + w]
            
            if face_bgr.size == 0:
                continue

            roi_gray = gray[y : y + h, x : x + w]

            face_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
            face_gray = cv2.resize(face_gray, (48, 48), interpolation = cv2.INTER_AREA)
            face_input = cv2.cvtColor(face_gray, cv2.COLOR_GRAY2BGR)

            try:
                pred = recognizer.predict(face_input)

                if pred is None or len(pred) < 2:
                    continue

                raw_emotion, confidence = pred
                confidence = float(confidence)

                print("RAW:", raw_emotion)

                emotion_history.append(raw_emotion)
                if len(emotion_history) > EMOTION_WINDOW:
                    emotion_history.pop(0)

                display_emotion = max(set(emotion_history), key=emotion_history.count)

                global angry_frame_count

                if raw_emotion.lower() == "angry":
                    angry_frame_count += 1
                else:
                    angry_frame_count = 0

                current_state = "angry" if angry_frame_count >= 5 else "calm"

                transition_detected = False
                if last_state == "calm" and current_state == "angry":
                    transition_detected = True

                last_state = current_state

                results.append((display_emotion, confidence))

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
                cv2.putText(frame, display_emotion, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

                if transition_detected:
                    cv2.putText(frame, "ESCALATION DETECTED",
                    (x, y - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0,0,255), 2)
            
            except Exception:
                continue

        return frame, results
    except Exception as e:
        print("Error in get_frame:", e)
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        return blank, []

# Release the camera resource
def release():
    global cap
    if cap is not None:
        cap.release()
        cap = None