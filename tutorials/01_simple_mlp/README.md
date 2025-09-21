#### EVERYTHING IN THIS DIRECTORY IS ALREADY GENERATED EXCEPT BINARIES/EXECUTABLES:
1. *simple_mlp.py* is the python script used to build a simple mlp using the sequential model class. The models input and output was normalized using the standard deviation and mean.
2. Run `python3 simple_mlp.py` in the terminal to run the python script.
3. The python script will generate *simple_mlp.h5* which is the saved neural net. It will also generate the input and output normalization parameters in *.npy* files that will be also used to code generate the model.
4. *read_each_layer.py* is then used to produce each layers output in a csv file from *simple_mlp.h5* and is saved to *layer_outputs/*.
5. Run 'python3 read_each_layer.py` in the terminal to produce each layers output (again already generated).
6. **CodeJeNN** is then used to generate *simple_mlp.hpp*.
7. Run `./generate.sh` to code generate the trained model. Note: the main.py path is in relation to where this directory is directly located in the github repo.
8. *test.cpp* was generated along with *simple_mlp.hpp* and is used to perform inference for *simple_mlp.hpp* that is then compared to the last output layer in *layer_outputs/*.
9. Run `clang++ simple_mlp.cpp` or `g++ simple_mlp.cpp` to create the binary for the test script to test the inference of the model. 
10. Finally, run `./a.out` in the terminal. You should get 10 values as the output and compare that with the output from *read_each_layer.py*.
11. If you want to try it yourself, you may run `./clean.sh` and restart from step 2.