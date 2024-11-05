import numpy as np
from keras.models import load_model

# Load the model
model = load_model('dump_model/test_model_1.h5')

# Normalization parameters from the .dat file
input_min = np.array([0.012154474689816341, 0.004632023004602859, 0.005522117123602399, 0.014544665667881929, 0.015456616528867428, 0.009197051616629648, 0.011353644767419069, 0.01083765148029836, 0.005061583846218687, 0.04086861626647886])
input_max = np.array([0.9905051420006733, 0.9997176732861306, 0.9966368370739054, 0.9626484146779251, 0.9856504541106007, 0.9929647961193003, 0.9860010638228709, 0.9968742518459474, 0.9872761293149445, 0.9868869366005173])
output_min = np.array([0.029973589872677953, 0.004939980934409616, 0.022123551528997254, 0.0057586604981215705, 0.011620539908100636])
output_max = np.array([0.9713950940416396, 0.987668007996647, 0.9866625932671464, 0.9963339160567419, 0.9994137257706666])

# Input data for prediction (replace with your input)
input_data = np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])  # Adjust values and shape as per your requirements

# Normalize input data
input_data_normalized = (input_data - input_min) / (input_max - input_min)

# Make prediction
prediction_normalized = model.predict(input_data_normalized)

# Denormalize output
prediction = prediction_normalized * (output_max - output_min) + output_min

# Print prediction
print(prediction)
