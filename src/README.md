<!-- 
Distribution Statement A. Approved for public release, distribution is unlimited.
---
THIS SOURCE CODE IS UNDER THE CUSTODY AND ADMINISTRATION OF THE GOVERNMENT OF THE UNITED STATES OF AMERICA.
BY USING, MODIFYING, OR DISSEMINATING THIS SOURCE CODE, YOU ACCEPT THE TERMS AND CONDITIONS IN THE NRL OPEN LICENSE AGREEMENT.
USE, MODIFICATION, AND DISSEMINATION ARE PERMITTED ONLY IN ACCORDANCE WITH THE TERMS AND CONDITIONS OF THE NRL OPEN LICENSE AGREEMENT.
NO OTHER RIGHTS OR LICENSES ARE GRANTED. UNAUTHORIZED USE, SALE, CONVEYANCE, DISPOSITION, OR MODIFICATION OF THIS SOURCE CODE
MAY RESULT IN CIVIL PENALTIES AND/OR CRIMINAL PENALTIES UNDER 18 U.S.C. § 641.
-->

# Directory Contents
  * **codegen/** ⮕ Folder that holds all methods/function to extract, load, and code generate any model in a directory where your neural nets are stored.

  * **generate.sh** ⮕ Bash script you can use if you wish to generate c++ files, or include the contents in your own bash script or makefile.

  * **dump_model/** ⮕ Default dump folder that you can dump any trained model and link during the code generation process (you can link another directory if you want).

  * **bin/** ⮕ Default folder where genereated header files are created (you can link another directory if you want).

  * **testing/** ⮕ Folder to compare and contrast header file to keras' output per layer.

# Code Generation Explanation and Limiations

### File Type and Directory:

* CodeJeNN code generations a train neural net stored on a **.keras**,a **.h5** file. Tensorflow keras was chosen because of its portability. Because the NN parameters, hyperparameters and the architecture itself is stored on the file as opposed to PyTorch, CodeJeNN only needs that file.

* You link the input folder with all trained models (infinite amount of **.keras** or **.h5** files if you please) wanting to be code generated as well as the output folder to save the files. Additionally, the data type precision is optional, but only accepts float or double.

    * **dump_model/** is the default linked dump folder to place trained models.

* The generated header file and "predict" function will be named after the file name of the trained model that it was code generated from.

    * Example: ***my_model.h5***  will code generate the header file  ***my_model.h*** and the predict function template will become ***auto my_model(nn_inputs)***.

    * Along with that, a **test.cpp** will also be copied into the desired directory. This file makes sure logic and accuracy of the trained neural net matches with results during training. 

### Standardization/Normalization:

* CodeJeNN supports normalization and standardization of inputs and outputs. 

    * The standardization/normalization values must be stored in a **.txt**, **.dat**, or a **.csv** file with **SPACES AS DELIMITERS**.

    * The normalization/standardization file name must match the name of the **.keras**, **.h5**, or **.onnx** file to make sure the proper parameters are generated correctly.

    * A full example OF HOW TO PROPERLY NAME THE NORMALIZATION PARAMETERS is in the **example.dat** file in **dump_model/** or in the **../tutorials/** directory. The important thing to remember is to use commas as delimiters, use the correct variable and file name, and put all array of values in brackets.

### Model Architecture

* CodeJeNN suports CNNs and MLPs. CodeJeNN only supports single input and single output sequential models or stacked layer models, more information can be found in the **../tutorial/** directory. Single input and single output refers to a scalar, an array, a 2d matrix or 3d matrix. 

    * If the user wants to train a nueral net such that the input is a matrix A and a vector b to predict x such as solving Ax=b, CodeJeNN does not have the capabilites to codegenerate that.

    * Sequential models or stacked layer models referes to how the user can build an MLP or CNN using tensorflow keras. You can stack different types of layers on top whether they are Dense, Conv2D, Activations, Normalization layers and etc. 

* Unfortunately, at the time of writing, tensorflow keras is currently implementing API for GNNs.

* To truly understand how complex and deep these architectures can be code generated, the user is referred to the **../tutorial/** directory.

* Once the model is code generated, the header file structure is as follows (percentage denotes space taken in the header file):
    1. import std libraries **>1%**
    1. layer propagation functions (Dense(), UnitNormalization(), Conv2DTranspose, etc...) **~10%**
    1. function header template **>1%**
    1. arrays of layer parameters (weights, baises, strides, kernels, gamma, etc...) **~85%**
    1. functions calls to drive through the each layer of the model. **~5%**

### Aglorithm Complexity

* CodeJeNN generates c++ interpreted nueral nets for computaional physics application, which means these NN are used within big codes. CodeJeNN takes advantage of what c++ has to offer in terms of runtime calculation speed. 

* It is taboo to use vectorization or parallelization within the c++ NN because outside of that, the code is assumed to be already vectorized and parallelized thus no threads are left and implementation of such actions can result in slower inference and slower code overall.

* However, CodeJeNN tries to inline as much as possible by using templating, lambda functions, less cache spaced used, etc.

    * Constexpr static arrays are used as much as possible and variables is passsed by constant reference or by pointer as much as possible to reduce memory usage. 

    * For Loops can be optimized using variadic templating which inlines and unravels the code, increasing compile time and cache space used, but significantly decreases run time. CodeJeNN sticks to for loops because of how long and extensive these weight and biases arrays can get. 

    * The layer propagation functions such as Conv1D(), Conv2D(), Dense(), LayerNormalization(), are inlined as much as possible and memroy is allocated beforehand in these functions as much as possible.

    * Activation functions are not considered layer propagation functions and are defined and lambda functions to inline as much as possible. In the layer propagation function template headers, the activation function is defined as a template parameter thus the activation functions are passed by lambda and inlined as much as possible. 

* The user is also encouraged to optimize the code generated neural net as much as possible too. 

### Error Handling

* CodeJeNN is not perfect. Thus, numerous errors (i.e. try catch blocks) are used to help the user zero in on the code generation errors. From testing, most of the time, errors are due to the in-capabilites of CodeJeNN.

    * If such errors occurs, the user is very, and I mean very encouraged to add and update the logic and submit a pull request and merge. 

# Code Generation Steps
1. First create trained neural nets using tensorflow keras and save in the supported file extensions.

    * If necessary, save normalization/standardization parameters in a **.dat**, a **.csv**, or a **.txt** file format. 

1. Put all models wanting to be code generated in the **dump_model/** folder, or whatever directory you wish to link.

1. Link necessary directories or change the bash script **(generate.sh)** dependicies as stated below.

    1. `--input` ⮕ path to folder that holds any and all trained models to be code generated, or use **dump_model/**.

    1. `--output` ⮕ path to folder to save all generated header files, or use **generated_model/**.

    1. `--precision` ⮕ (OPTIONAL) variable type of precision, either double or float. If not specified, will default to float.

        ```bash
        python ./codegen/main.py --input="./dump_model" --output="./bin" --precision="double"
        ```
1. Run **generate.sh** (type `bash generate.sh` in terminal/shell).

1. Once generation is complete. You are done!

# Testing Generated Predict Function

* As said before, a test.cpp file will also be copied into the output directory so the user can compare the c++ NN output with the python tensorflow keras output. 

* The user may edit the **test.cpp** source file to check for precision and accuracy of trained neural net. Down below is what needs to be change in **test.cpp**

    **Header File** 
    ```c++
    // change file name to desired header file
    #include "header_file.h"
    ```
    **Input**
    ```c++
    // change input to desired features
    std::array<Scalar, "number_of_input_features"> input = {"input(s)"};
    auto output = "function_name"<Scalar>(input);
    ```
1. Once done, compile code. Comments on the preferred method to compile and run **test.cpp** is at the bottom of the source file.

1. Verify that the output is correct!

# Try an example already in **dump_model/**
HOPEFULLY YOU READ ALL THIS, you can now try out the example in **dump_model/**. Just open a terminal/shell in the **src/** directory, KEEP ONE OF THE OPTIONS FOR NORMALIZATION/STANDARDIZATION IN `example.dat` AND DELETE THE REST, link the correct folders in **generate.sh**, type `bash generate.sh` in the terminal/shell, and you are good to go!