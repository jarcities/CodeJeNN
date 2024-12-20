## File Stucture
  * **model_dump** -> Dump folder for model files holding neural net structure.
  * **logic_tests** -> Source files with tests cases to validate the logic of the neural net methods **model_methods.h** **model_methods.cpp**
  * **main.py** -> Main python script that runs code generation given the h5 file. 
  * **model_methods.cpp** -> Source file with neural net methods definition.
  * **model_methods.h** -> Header file with neural net methods declaration.

## Compilation Requirements
Libraries:
```zsh
pip install onnx onnxruntime tf2onnx torch tensorflow keras2onnx onnx2keras h5py 
```

or 

```zsh
pip3 install onnx onnxruntime tf2onnx torch tensorflow keras2onnx onnx2keras h5py 
```

Compiler Check:
```sh
/usr/bin/clang++ --version
/usr/bin/lldb --version
```

Debug Configuration:
```sh
clang++ - Build and Debug Active File
```

## Compilation Steps

1. Drop any HDF5 (h5) files in **model_dump**.

1. Run **main.py** first.

1. Terminal will prompt user for two things:
    * Which h5 file to use.
    * Which initial inputs to use.
    
    Input the correct file name and correct inputs.

1. Open new terminal session and `cd` to **Main**.

1. Terminal Commands:

    > **For main source file**:
    >
    > `clang++ -std=c++14 -o main main.cpp model_methods.cpp`
    >
    > `./main`

    > **For logic tests**:
    >
    > `clang++ -std=c++14 -o <test_file_name> logic_tests/<test_file_name>.cpp model_methods.cpp`
    >
    > `./<test_file_name>`

#### AFTER EVERY COMPILATION DELETE ALL UN-ORIGINAL FILES.

## Logic Structure

  1. Read in NN file based on the user input.
  1. Load that h5 files neural net data into **model**.
  1. Create python arrays of weights, biases, activation functions, alphas, and dropout rates.
      * Special case applied for **Leaky ReLU** and **ELU**
  1. Iterate through each layer using the **model.layers** from keras as the iterative method. 
      1. We retrieve the weights, bias, activation function, alpha value, and dropout rate from each layer
      1. Put all that information into the python arrays that were alreayd initialized. 
  1. Now that we have all the information, transfer all that information to c++ (codegen).
      1. The c++ code should not have any for loops, it will go through each layer one by one.
      1. Create as many arrays of weights and biases per layer. 
      1. Create as many **foward_propagation** methods as necessary depening on how many layers there are. 
          * All neural net methods are already created and abstracted away.
  1. Save all that into **main.cpp**.

## Makefile

If **Makefile** is necessary, then copy syntax below:

```makefile
CXX = clang++
CXXFLAGS = -std=c++14 -Wall

all: main

main: main.o model_methods.o
	$(CXX) $(CXXFLAGS) -o main main.o model_methods.o

main.o: main.cpp model_methods.h
	$(CXX) $(CXXFLAGS) -c main.cpp

model_methods.o: model_methods.cpp model_methods.h
	$(CXX) $(CXXFLAGS) -c model_methods.cpp

clean:
	rm -f main main.o model_methods.o
```