1/ cnn2.py was used to build a test convolutional neural net.
2/ cnn2.keras is the neural net that was built.
3/ read_each_layer.py is then used to produce each output of cnn2.keras and prints csv files to /layer_outputs.
4/ codejenn is then used to generate cnn2.h.
5/ test.cpp was generated along with cnn2.h and is used to perform inference for cnn2.h that is then compared to the last output layer in /layer_outputs.