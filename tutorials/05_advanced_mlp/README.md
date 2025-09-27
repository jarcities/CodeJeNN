#### EVERYTHING IN THIS DIRECTORY IS ALREADY GENERATED EXCEPT BINARIES/EXECUTABLES:

1. *advanced_mlp.py* is the python script used to build an advanced mlp using the sequential model class. The models input and output was normalized using the standard deviation and mean. Run `python3 advanced_mlp.py` in the terminal to run the python script.

1. The python script will generate *advanced_mlp.h5* which is the saved neural net. It will also generate the input and output normalization parameters in *.npy* files that will be also used to code generate the model.

1. *read_each_layer.py* uses keras python API to produce each layers output in a csv file from *advanced_mlp.h5* and is saved to *layer_outputs/*. Run `python3 read_each_layer.py` in the terminal to produce each layers output (again already generated).

1. This specific tutorial has the `--custom_activation` flag on.

    1. First, the user trains with the custom activation as shown in `advanced_mlp.py`.

    1. Next, the user defines the custom activation in another python script named after the activation function, i.e. `custom_activation.py`. 
    
        In this custom activation python script, the user must serialize it using `@register_keras_serializable()`.

1. **CodeJeNN** then generates ***advanced_mlp.hpp***. Run `./generate.sh` to code generate the trained model. Note: the main.py path is in relation to where this directory is directly located in the github repo.

1. Because a custom activation function was used, the user must edit ***advanced_mlp.hpp*** by adding their C++ rendition of the custom activation function. Print statements are in place if the user forgets to do so during inference.

1. ***test.cpp*** was generated along with ***advanced_mlp.hpp*** and is used to perform inference for ***advanced_mlp.hpp*** that is then compared to the last output layer in ***layer_outputs/***.

1. Run `clang++ advanced_mlp.cpp` or `g++ advanced_mlp.cpp` to create the binary for the test script to test the inference of the model. 

1. Finally, run `./a.out` in the terminal. You should get 100 values as the output and compare that with the output from ***read_each_layer.py***.

1. This specific tutorial also has the `--debug` flag on so you may see how the debug feature works. 

    1. First when running `python3 generate.sh` it will print the `model.summary` of the neural net from tensorflow keras, and will print what **CodeJeNN** extracted for comparison. 

    1. Then, within the header file, will also be print statements to see the output of each layer. Running `clang++ advanced_mlp.cpp` or `g++ advanced_mlp.cpp` will print each layers output.

1. If you want to try it yourself, you may run `./clean.sh` and restart from step 1.