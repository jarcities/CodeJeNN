#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

def main():
    # Load the model saved in cnn6.h5
    model = load_model('cnn6.h5')
    
    # Print a summary of the model architecture
    print("Model Summary:")
    model.summary()

    # Iterate over layers to print their output shapes
    print("\nLayer-wise Output Shapes:")
    for layer in model.layers:
        try:
            output_shape = layer.output_shape
        except AttributeError:
            output_shape = "Not available"
        layer_type = layer.__class__.__name__
        print(f"Layer Name: {layer.name} | Type: {layer_type} | Output Shape: {output_shape}")

        # If this layer is a Conv2DTranspose, highlight its shape.
        if isinstance(layer, tf.keras.layers.Conv2DTranspose):
            print(" --> Conv2DTranspose layer found!")
            print(f"     [Conv2DTranspose Output Shape]: {output_shape}\n")
    
if __name__ == '__main__':
    main()
