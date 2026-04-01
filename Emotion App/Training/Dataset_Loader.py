# This imports necessary libraries for loading and preprocessing an image dataset for emotion recognition.
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define a class to load and preprocess the emotion dataset
class Emotion_Dataset_Loader:
    # Initialize the dataset loader
    # dataset_path: path to the dataset directory
    # img_size: target size for the images (default is 64x64)
    # batch_size: number of images to be processed in a batch (default is 32)
    def __init__(self, dataset_path, img_size=(64, 64), batch_size=32):
        # The image data generator handles:
        # pixel normalisation
        # automatic splitting of the dataset into training and validation sets
        self.datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.batch_size = batch_size

    # Method to load training data (80% of the dataset)
    def train_data(self):
        return self.datagen.flow_from_directory(
            self.dataset_path,
            target_size=self.img_size, # Resize images to the specified target size
            class_mode='categorical', # Use categorical labels for multi-class classification
            subset='training'
        )

    # Method to load validation data (20% of the dataset)
    def val_data(self):
        return self.datagen.flow_from_directory(
            self.dataset_path,
            target_size=self.img_size,
            class_mode='categorical',
            subset='validation'
        )