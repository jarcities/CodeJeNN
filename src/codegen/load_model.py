# Distribution Statement A. Approved for public release, distribution is unlimited.
"""
THIS SOURCE CODE IS UNDER THE CUSTODY AND ADMINISTRATION OF THE GOVERNMENT OF THE UNITED STATES OF AMERICA.
BY USING, MODIFYING, OR DISSEMINATING THIS SOURCE CODE, YOU ACCEPT THE TERMS AND CONDITIONS IN THE NRL OPEN LICENSE AGREEMENT.
USE, MODIFICATION, AND DISSEMINATION ARE PERMITTED ONLY IN ACCORDANCE WITH THE TERMS AND CONDITIONS OF THE NRL OPEN LICENSE AGREEMENT.
NO OTHER RIGHTS OR LICENSES ARE GRANTED. UNAUTHORIZED USE, SALE, CONVEYANCE, DISPOSITION, OR MODIFICATION OF THIS SOURCE CODE
MAY RESULT IN CIVIL PENALTIES AND/OR CRIMINAL PENALTIES UNDER 18 U.S.C. § 641.
"""

import os
import tensorflow as tf
import onnx
from tensorflow.python.keras.models import load_model as lm
from keras.models import load_model


def loadModel(file_path):
    """
    Load a model from the given file path.

    This function supports loading models with the following extensions:
    - .h5: Keras HDF5 model file
    - .keras: Keras model file
    - SavedModel: TensorFlow SavedModel directory
    - .onnx: ONNX model file

    Args:
        file_path (str): The path to the model file.

    Returns:
        tuple: A tuple containing the loaded model and the file extension.

    Raises:
        ValueError: If the file type is unsupported or if there is an error loading the model.
    """
    # Extract the file name and extension from the file path
    file_name, file_extension = os.path.splitext(file_path)

    # Check if the file extension is .h5 or .keras
    if file_extension == ".h5" or file_extension == ".keras":
        # Define custom objects for loading the model, specifically for LeakyReLU layers
        custom_objects = {"LeakyReLU": tf.keras.layers.LeakyReLU}
        # Initialize an empty list to store any errors that occur during loading
        errors = []
        # Attempt to load the model using tensorflow.keras.models.load_model
        try:
            model = lm(file_path)
        except Exception as e:
            # If an error occurs, append it to the errors list
            errors.append(
                f"\nError loading model from {file_name} with tensorflow.keras: {e}\n"
            )
        # If the model was not successfully loaded, attempt to load it using keras.models.load_model
        if "model" not in locals():
            try:
                # Attempt to load the model without custom objects
                model = load_model(file_path)
            except Exception as e:
                # If an error occurs, append it to the errors list
                errors.append(
                    f"\nError loading model from {file_name} with load_model using keras (no custom_objects): {e}\n"
                )
        # If the model was still not successfully loaded, attempt to load it using keras.models.load_model with custom objects
        if "model" not in locals():
            try:
                # Attempt to load the model with custom objects
                model = load_model(file_path, custom_objects=custom_objects)
            except Exception as e:
                # If an error occurs, append it to the errors list
                errors.append(
                    f"\nError loading model from {file_name} with load_model using keras (with custom_objects): {e}\n"
                )
        # If the model was still not successfully loaded, attempt to load it using keras.models.load_model without compiling
        if "model" not in locals():
            try:
                # Attempt to load the model without compiling
                model = load_model(file_path, compile=False)
            except Exception as e:
                # If an error occurs, append it to the errors list
                errors.append(
                    f"\nError loading model from {file_name} with load_model using keras (no compiling): {e}\n"
                )
        # If all attempts to load the model have failed
        if "model" not in locals():
            # Join all the error messages into a single string
            error_message = "\n".join(errors)
            # Print the aggregated error message
            print(f"\nAll attempts to load the model failed:\n{error_message}\n")

    # Check if the file extension is .onnx
    elif file_extension == ".onnx":
        # Attempt to load the ONNX model
        try:
            model = onnx.load(file_path)
        except Exception as e:
            # If an error occurs, raise a ValueError with the error message
            raise ValueError(f"\nError loading ONNX model from {file_name}: {e}\n")

    # If the file extension is not supported
    else:
        # Raise a ValueError indicating that the file type is not supported
        raise ValueError("\nUnsupported file type\n")

    # Return the loaded model and the file extension
    return model, file_extension
