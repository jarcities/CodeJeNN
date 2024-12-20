# Distribution Statement A. Approved for public release, distribution is unlimited.
"""
THIS SOURCE CODE IS UNDER THE CUSTODY AND ADMINISTRATION OF THE GOVERNMENT OF THE UNITED STATES OF AMERICA. BY USING, MODIFYING, OR DISSEMINATING THIS SOURCE CODE, YOU ACCEPT THE TERMS AND CONDITIONS IN THE NRL OPEN LICENSE AGREEMENT. USE, MODIFICATION, AND DISSEMINATION ARE PERMITTED ONLY IN ACCORDANCE WITH THE TERMS AND CONDITIONS OF THE NRL OPEN LICENSE AGREEMENT. NO OTHER RIGHTS OR LICENSES ARE GRANTED. UNAUTHORIZED USE, SALE, CONVEYANCE, DISPOSITION, OR MODIFICATION OF THIS SOURCE CODE MAY RESULT IN CIVIL PENALTIES AND/OR CRIMINAL PENALTIES UNDER 18 U.S.C. § 641.
"""

import os
import tensorflow as tf
import onnx
from tensorflow.python.keras.models import load_model as lm
from keras.models import load_model


def loadModel(file_path):
    """
    load model from file path based on file extension
    supports .h5, .keras, SavedModel, .onnx
    """
    file_name, file_extension = os.path.splitext(file_path)

    if file_extension == '.h5' or file_extension == '.keras':
        custom_objects={'LeakyReLU':tf.keras.layers.LeakyReLU}
        errors=[]
        try:
            model=lm(file_path)
        except Exception as e:
            errors.append(f"\nError loading model from {file_name} with tensorflow.keras: {e}\n")
        if 'model' not in locals():
            try:
                model=load_model(file_path)
            except Exception as e:
                errors.append(f"\nError loading model from {file_name} with load_model using keras (no custom_objects): {e}\n")
        if 'model' not in locals():
            try:
                model=load_model(file_path,custom_objects=custom_objects)
            except Exception as e:
                errors.append(f"\nError loading model from {file_name} with load_model using keras (with custom_objects): {e}\n")
        if 'model' not in locals():
            try:
                model=load_model(file_path,compile=False)
            except Exception as e:
                errors.append(f"\nError loading model from {file_name} with load_model using keras (no compiling): {e}\n")
        if 'model' not in locals():
            error_message="\n".join(errors)
            print(f"\nAll attempts to load the model failed:\n{error_message}\n")

    elif file_extension == '.onnx':
        try:
            model = onnx.load(file_path)
        except Exception as e:
            raise ValueError(f"\nError loading ONNX model from {file_name}: {e}\n")
        
    else:
        raise ValueError("\nUnsupported file type\n")

    return model, file_extension