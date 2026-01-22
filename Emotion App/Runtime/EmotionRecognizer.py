import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Define a class for emotion recognition using a pre-trained model
class EmotionRecognizerModel:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        # Define the list of emotions
        self.emotions = [
            "angry",
            "neutral",
            "happy",            
        ]
    # Preprocess the input face image for prediction
    def preprocess(self, face):
        face = cv2.resize(face, (64, 64))
        face = face.astype("float32") / 255.0
        return np.expand_dims(face, axis=0)

    # Predict the emotion of the given face image
    def predict(self, face):
        preds = self.model.predict(self.preprocess(face), verbose=0)
        idx = np.argmax(preds)
        return self.emotions[idx], float(preds[0][idx])

    # Example usage:bxdsqlcubdeqi