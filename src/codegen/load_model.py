
import os
import tensorflow as tf
import onnx
from tensorflow.python.keras.models import load_model as lm
from keras.models import load_model


def loadModel(file_path):
    # Section: Extract File Extension
    file_name, file_extension = os.path.splitext(file_path)

    # Section: Load Keras Model (.h5 or .keras)
    if file_extension == ".h5" or file_extension == ".keras":
        custom_objects = {"LeakyReLU": tf.keras.layers.LeakyReLU}
        errors = []
        try:
            model = lm(file_path)
        except Exception as e:
            errors.append(f"\nError loading model from {file_name} with tensorflow.keras: {e}\n")
        if "model" not in locals():
            try:
                model = load_model(file_path)
            except Exception as e:
                errors.append(f"\nError loading model from {file_name} with load_model using keras (no custom_objects): {e}\n")
        if "model" not in locals():
            try:
                model = load_model(file_path, custom_objects=custom_objects)
            except Exception as e:
                errors.append(f"\nError loading model from {file_name} with load_model using keras (with custom_objects): {e}\n")
        if "model" not in locals():
            try:
                model = load_model(file_path, compile=False)
            except Exception as e:
                errors.append(f"\nError loading model from {file_name} with load_model using keras (no compiling): {e}\n")
        if "model" not in locals():
            error_message = "\n".join(errors)
            print(f"\nAll attempts to load the model failed:\n{error_message}\n")

    # Section: Load ONNX Model (.onnx)
    elif file_extension == ".onnx":
        try:
            model = onnx.load(file_path)
        except Exception as e:
            raise ValueError(f"\nError loading ONNX model from {file_name}: {e}\n")

    # Section: Unsupported File Type
    else:
        raise ValueError("\nUnsupported file type\n")

    return model, file_extension
