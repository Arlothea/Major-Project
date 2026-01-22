# This imports necessary libraries for loading and preprocessing an image dataset for emotion recognition.
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define a class to load and preprocess the emotion dataset
class Emotion_Dataset_Loader:
    # Initialize the dataset loader with dataset path, image size, and batch size
    def __init__(self, dataset_path, img_size=(64, 64), batch_size=32):
        self.datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.batch_size = batch_size

    # Method to load training data
    def train_data(self):
        return self.datagen.flow_from_directory(
            self.dataset_path,
            target_size=self.img_size,
            class_mode='categorical',
            subset='training'
        )

    # Method to load validation data
    def val_data(self):
        return self.datagen.flow_from_directory(
            self.dataset_path,
            target_size=self.img_size,
            class_mode='categorical',
            subset='validation'
        )




