#### EVERYTHING IN THIS DIRECTORY IS ALREADY GENERATED EXCEPT BINARIES/EXECUTABLES:
1. *advanced_mlp.py* is the python script used to build an advanced mlp using the sequential model class. The models input and output was normalized using the standard deviation and mean. Run `python3 advanced_mlp.py` in the terminal to run the python script.
3. The python script will generate *advanced_mlp.h5* which is the saved neural net. It will also generate the input and output normalization parameters in *.npy* files that will be also used to code generate the model.
4. *read_each_layer.py* uses keras python API to produce each layers output in a csv file from *advanced_mlp.h5* and is saved to *layer_outputs/*. Run `python3 read_each_layer.py` in the terminal to produce each layers output (again already generated).
6. **CodeJeNN** is then generate *advanced_mlp.hpp*. Run `./generate.sh` to code generate the trained model. Note: the main.py path is in relation to where this directory is directly located in the github repo.
8. *test.cpp* was generated along with *advanced_mlp.hpp* and is used to perform inference for *advanced_mlp.hpp* that is then compared to the last output layer in *layer_outputs/*.
9. Run `clang++ advanced_mlp.cpp` or `g++ advanced_mlp.cpp` to create the binary for the test script to test the inference of the model. 
10. Finally, run `./a.out` in the terminal. You should get 100 values as the output and compare that with the output from *read_each_layer.py*.
11. This specific tutorial has the `--debug` flag on so you may see how the debug feature works. 
11. If you want to try it yourself, you may run `./clean.sh` and restart from step 1.