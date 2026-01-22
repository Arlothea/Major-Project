# Emotion Recognition Model using TensorFlow/Keras
from pickletools import optimize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define the EmotionModel class
class EmotionModel:
    # Initialize the model with input shape and number of emotion classes
    def __init__(self, input_shape=(64, 64, 3), num_emotion=3):
        self.model = self._build_model(input_shape, num_emotion)
    
    # Build the CNN model architecture
    def _build_model(self, input_shape, num_emotion):
        model = Sequential([
            # First convolutional layer
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D(2, 2),
            
            # Second convolutional layer
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(num_emotion, activation='softmax')
        ])

        # Compile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    # Train the model with training and validation data
    def train(self, train_data, val_data, epochs=50):
        self.model.fit(train_data, validation_data=val_data, epochs=epochs)

    # Save the trained model to the specified path
    def save(self, path):
        self.model.save(path)




