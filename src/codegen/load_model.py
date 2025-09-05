
# import os
# import tensorflow as tf
# from tensorflow.python.keras.models import load_model as lm
# from keras.models import load_model


# def loadModel(file_path):

#     # split file type and file name
#     file_name, file_extension = os.path.splitext(file_path)

#     # load keras model based on version
#     if file_extension == ".h5" or file_extension == ".keras":
#         custom_objects = {"LeakyReLU": tf.keras.layers.LeakyReLU}
#         errors = []
#         try:
#             model = lm(file_path, compile=False)
#         except Exception as e:
#             errors.append(f"\nError loading model from {file_name} with tensorflow.keras: {e}\n")
#         if "model" not in locals():
#             try:
#                 model = load_model(file_path, compile=False)
#             except Exception as e:
#                 errors.append(f"\nError loading model from {file_name}, (no custom_objects, no compiling): {e}\n")
#         if "model" not in locals():
#             try:
#                 model = load_model(file_path, custom_objects=custom_objects, compile=False)
#             except Exception as e:
#                 errors.append(f"\nError loading model from {file_name}, (with custom_objects, no compiling): {e}\n")
#         if "model" not in locals():
#             error_message = "\n".join(errors)
#             print(f"\nAll attempts to load the model failed:\n{error_message}\n")

#     else:
#         raise ValueError("\nUnsupported file type\n")

#     return model, file_extension



















































from email.mime import base
import os
import tensorflow as tf
from tensorflow.python.keras.models import load_model as tf_load_model
from keras.models import load_model as keras_load_model


def loadModel(file_path, base_file_name, user_activation):
    # ==============================================================
    # load a keras model from .h5 or .keras using keras and tf with 
    # combos of compile and custom objects options.
    # ==============================================================
    file_name, file_extension = os.path.splitext(file_path)
    if file_extension not in (".h5", ".keras"):
        raise ValueError(f"Unsupported file type: {file_extension}")

    #################################################################
    #custom activation function handling
    activation_name = user_activation   
    if activation_name:
        import sys
        sys.path.append(os.path.abspath("dump_model"))
        import importlib
        module = importlib.import_module(base_file_name)
        act_fun = getattr(module, activation_name)
        act_fun = act_fun.__name__
    else:
        act_fun = None
    custom_objects = {
        "LeakyReLU": tf.keras.layers.LeakyReLU
    }
    if act_fun:
        custom_objects[activation_name] = act_fun
    #################################################################

    errors = []

    #try 1
    try:
        model = tf_load_model(
            file_path,
            custom_objects=custom_objects,
            compile=False
        )
        return model, file_extension
    except Exception as e:
        errors.append(f"TF-Keras compile=False: {e}")

    #try 2
    try:
        model = keras_load_model(
            file_path,
            custom_objects=custom_objects,
            compile=False
        )
        return model, file_extension
    except Exception as e:
        errors.append(f"Keras compile=False, custom_objects: {e}")

    #try 3
    try:
        model = keras_load_model(
            file_path,
            compile=False
        )
        return model, file_extension
    except Exception as e:
        errors.append(f"Keras compile=False, no custom_objects: {e}")

    #try 4
    try:
        model = tf_load_model(
            file_path,
            custom_objects=custom_objects
        )
        return model, file_extension
    except Exception as e:
        errors.append(f"TF-Keras default compile: {e}")

    #try 5
    try:
        model = keras_load_model(
            file_path,
            custom_objects=custom_objects
        )
        return model, file_extension
    except Exception as e:
        errors.append(f"Keras default compile, custom_objects: {e}")

    #try 6
    try:
        model = keras_load_model(
            file_path
        )
        return model, file_extension
    except Exception as e:
        errors.append(f"Keras default compile, no custom_objects: {e}")

    #all is lost
    error_message = "\n".join(errors)
    raise RuntimeError(
        f"All attempts to load the model failed for {file_name}:\n{error_message}"
    )
