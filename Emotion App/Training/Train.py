# Imports the necessary loader module
from Dataset_Loader import Emotion_Dataset_Loader
# Import the EmotionModel class
from Emotion_Model import EmotionModel
import os
# Define dataset and model paths
DATASET_PATH = "../dataset"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "Models", "emotion_model.h5")


# Load dataset
loader = Emotion_Dataset_Loader(DATASET_PATH)

# Get training and validation data
train_data = loader.train_data()
val_data = loader.val_data()

# Initialize and train the emotion recognition model
model = EmotionModel(
    input_shape=(64, 64, 3),
    num_emotion=train_data.num_classes
)

# Train the model and save it to the specified path
model.train(train_data, val_data, epochs=25)
model.save(MODEL_PATH)

print("Emotion model trained and saved")
