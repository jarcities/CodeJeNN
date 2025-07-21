from keras.models import load_model
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import get_custom_objects
from custom_activation import nonzero_diag_activation  # wherever you put it

get_custom_objects().update({
    'nonzero_diag_activation': nonzero_diag_activation
})

model = load_model('MLP_LU.keras', compile=False)
model.summary()
