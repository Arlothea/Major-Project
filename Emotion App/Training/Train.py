# Imports the necessary loader module
from Dataset_Loader import Emotion_Dataset_Loader
# Import the EmotionModel class
from Emotion_Model import EmotionModel
import os

# ------------- PATHS -------------
# Define dataset and model paths
DATASET_PATH = "../dataset"

# Builds the path to the model file, ensuring it is saved in the "Models" directory.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "Models", "emotion_model.h5")

# ---------------- DATASET LOADING ----------------
# Initialize the dataset loader with the specified dataset path
loader = Emotion_Dataset_Loader(DATASET_PATH)

# Get training and validation data
train_data = loader.train_data()
val_data = loader.val_data()

# ---------------- CREATE AND TRAIN THE MODEL ------------------
# Initialize and train the emotion recognition model
model = EmotionModel(
    input_shape=(64, 64, 3), # Expected image size and channels.
    num_emotion=train_data.num_classes # Number of emotion categories.
)

# ----------------- TRAIN MODEL ------------------
# Train the model using training data from within the dataset folder and validate using validation data.
model.train(train_data, val_data, epochs=50)

# -------------------- SAVE THE MODEL ------------------
# Save the trained model to the specified path, ensuring it is stored in the "Models" directory for future use.
model.save(MODEL_PATH)

print("Emotion model trained and saved")