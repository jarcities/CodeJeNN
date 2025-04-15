1/ dense5.py was used to build a test regular sequential neural net.
2/ dense5.h5 is the neural net that was built.
3/ read_each_layer.py is then used to produce each output of dense5.h5 and prints csv files to /layer_outputs.
4/ codejenn is then used to generate dense5.h.
5/ test.cpp was generated along with dense5.h and is used to perform inference for dense5.h that is then compared to the last output layer in /layer_outputs.