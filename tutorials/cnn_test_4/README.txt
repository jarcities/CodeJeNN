1/ cnn4.py was used to build a test convolutional neural net.
2/ cnn4.keras is the neural net that was built.
3/ read_each_layer.py is then used to produce each output of cnn4.keras and prints csv files to /layer_outputs.
4/ codejenn is then used to generate cnn4.h.
5/ test.cpp was generated along with cnn4.h and is used to perform inference for cnn4.h that is then compared to the last output layer in /layer_outputs.