#### EVERYTHING IN THIS DIRECTORY IS ALREADY GENERATED EXCEPT BINARIES/EXECUTABLES:
1. *cnn_2d.py* is the python script used to build a 2d convolutional neural net using the functional model class (similar to the 1d tutorial). The models input and output was normalized using the max and mean per column (input feature).
2. Run `python3 cnn_2d.py` in the terminal to run the python script that trains and saves the neural net.
3. The python script will generate *cnn_2d.keras* which is the saved convolutional neural net. It will also generate the input and output normalization parameters in *.npy* files that will be also used to code generate the model.
4. *read_each_layer.py* is then used to produce each layers output using keras' `predict` function from *cnn_2d.keras* and is then saved to *layer_outputs/*.
5. Run 'python3 read_each_layer.py` in the terminal to produce each layers output (again already generated).
6. **CodeJeNN** is then used to generate *cnn_2d.hpp*.
7. Run `./generate.sh` to code generate the trained model. Note: the main.py path is in relation to where this directory is directly located in the github repo.
8. *test.cpp* was generated along with *cnn_2d.hpp* and is used to perform inference for *cnn_2d.hpp* that is then compared to the last output layer in *layer_outputs/*.
9. Run `clang++ cnn_2d.cpp` or `g++ cnn_2d.cpp` to create the binary for the test script to test the inference of the model. 
10. Finally, run `./a.out` in the terminal. You should get 10 values as the output and compare that with the output from *read_each_layer.py*.
11. If you want to try it yourself, you may run `./clean.sh` and restart from step 2.