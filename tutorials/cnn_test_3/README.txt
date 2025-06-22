1/ cnn3.py was used to build a test convolutional neural net.
2/ cnn3.keras is the neural net that was built.
3/ read_each_layer.py is then used to produce each output of cnn3.keras and prints csv files to /layer_outputs.
4/ codejenn is then used to generate cnn3.h.
5/ test.cpp was generated along with cnn3.h and is used to perform inference for cnn3.h that is then compared to the last output layer in /layer_outputs.