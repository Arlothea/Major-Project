# Emotion Recognition Model using TensorFlow/Keras
from pickletools import optimize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# ---------------- CONVOLUTIONAL NEURAL NETWORK (CNN) -------------------- 
class EmotionModel:   
    # Initialize the model with input shape (64 x 64 RGB)
    # Defines the number of emotions (3: Happy, Neutral, Angry)
    def __init__(self, input_shape=(64, 64, 3), num_emotion=3):
        self.model = self._build_model(input_shape, num_emotion)
    
    # Build the CNN model architecture
    def _build_model(self, input_shape, num_emotion):
        model = Sequential([
            # First convolutional layer: Extract low-level features (edges, textures)
            # Applies 32 filters (3 x 3)
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D(2, 2),
            
            # Second convolutional layer: Extract higher-level facial features.
            # Uses 64 filters on more complex features
            # This detects emotional patterns such as eye tension, mouth curvature
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            
            # Flattens feature maps in 1D Vector
            Flatten(),
            # Connection layer
            # Dense learns combinations of the extracted features
            Dense(128, activation='relu'),
            # Dropout disables 50% of neurons during training which prevents overfitting and improves generalisation accuracy
            Dropout(0.5),
            # Output layer ( one neuron per emotion class)
            # Softmax converts outputs to probabilities
            # Highest probability = predicted emotion
            Dense(num_emotion, activation='softmax')
        ])

        # Compile the model with an optimiser using loss function and evaluation metric accuracy
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    # Train the model with training and validation data over 50 iterations (epochs)
    def train(self, train_data, val_data, epochs=50):
        self.model.fit(train_data, validation_data=val_data, epochs=epochs)

    # Save the trained model to the specified path
    def save(self, path):
        self.model.save(path)