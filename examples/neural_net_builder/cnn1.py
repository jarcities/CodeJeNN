import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Creates a simple CNN model using TensorFlow Keras.
    Args:
        input_shape (tuple): Shape of the input image, e.g. (28, 28, 1) for MNIST.
        num_classes (int): Number of output classes, e.g. 10 for digits 0-9.
    Returns:
        model (keras.Model): A compiled CNN model.
    """
    
    model = keras.Sequential([
        # Convolutional block 1
        layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', 
                      input_shape=input_shape),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Convolutional block 2
        layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten and Dense layers
        layers.Flatten(),
        layers.Dense(units=128, activation='relu'),
        
        # Output layer
        layers.Dense(units=num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.save("cnn1.h5")
    
    return model

if __name__ == '__main__':
    # Example usage with MNIST-sized grayscale images
    cnn_model = create_cnn_model(input_shape=(28, 28, 1), num_classes=10)
    cnn_model.summary()
