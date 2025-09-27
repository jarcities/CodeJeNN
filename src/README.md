<!-- 
Distribution Statement A. Approved for public release, distribution is unlimited.
---
THIS SOURCE CODE IS UNDER THE CUSTODY AND ADMINISTRATION OF THE GOVERNMENT OF THE UNITED STATES OF AMERICA.
BY USING, MODIFYING, OR DISSEMINATING THIS SOURCE CODE, YOU ACCEPT THE TERMS AND CONDITIONS IN THE NRL OPEN LICENSE AGREEMENT.
USE, MODIFICATION, AND DISSEMINATION ARE PERMITTED ONLY IN ACCORDANCE WITH THE TERMS AND CONDITIONS OF THE NRL OPEN LICENSE AGREEMENT.
NO OTHER RIGHTS OR LICENSES ARE GRANTED. UNAUTHORIZED USE, SALE, CONVEYANCE, DISPOSITION, OR MODIFICATION OF THIS SOURCE CODE
MAY RESULT IN CIVIL PENALTIES AND/OR CRIMINAL PENALTIES UNDER 18 U.S.C. § 641.
-->

## DIRECTORY CONTENTS
  * `api-core/` ⮕ Directory that holds all methods/function to extract, load, and code generate any model in a directory where your neural nets are stored.

  * `bin/` ⮕ Default directory where genereated header files are created (you can link another directory if you want).

  * `dump_model/` ⮕ Default dump directory that you can dump any trained model and link during the code generation process (you can link another directory if you want).

  * `generate.sh & clean.sh` ⮕ Default bash scripts used to generate and/or clean c++ files.

## CODE GENERATION STEPS
1. First create a trained neural nets using tensorflow keras and save in the supported file extensions `.h5` or `.keras`.

    * If necessary, save normalization/standardization parameters in a `.npy` file (example in `dump_model/`).

1. Put all models wanting to be code generated in the `dump_model/` directory, or whatever directory you wish to link.

1. Link necessary flags or change the bash script `generate.sh` as stated below:

    1. `--input` ⮕ path to directory that holds any and all trained models to be code generated, or use **dump_model/**.

    1. `--output` ⮕ path to directory to save all generated header files, or use **generated_model/**.

    1. `--precision` ⮕ (OPTIONAL) variable type of precision, either double or float. If not specified, will default to float.

    1. `--double` ⮕ (OPTIONAL) precision (float or double)

    1. `--float` ⮕ (OPTIONAL) precision (float or double)

    1. `--custom_activation` ⮕  (OPTIONAL)  if your model has a serialized custom activation function, attach the name of the activation function. (Example in `tutorials/05_advanced_mlp/`) 
    
        **NOTE**: The file name of the activation function must match the name of the activation function itself.

    1. `--debug` ⮕ (OPTIONAL) if there is inference errors with the trained neural net, you have the option to see what was extracted, and the output of each layer of the generated model. (Example in `tutorials/05_advanced_mlp/`)

    1. `--model_image` ⮕ (OPTIONAL) requires installing `graphviz` and `svn` allowing to see visual image of neural net.

        ```bash
        python3 \
            ./api-core/main.py \
            --input="./dump_model" \
            --output="./bin" \
            --double \
            --custom_activation="act_fun" \
            --debug \
            --model_image 
        ```

1. Run **generate.sh** (type `bash generate.sh` in terminal/shell).

1. Once generation is complete. You are done!

## About CodeJeNN (i)

* CodeJeNN code generations a train neural net stored on a **.keras**,a **.h5** file. Tensorflow keras was chosen because of its portability. Because the NN parameters, hyperparameters and the architecture itself is stored on the file as opposed to PyTorch, CodeJeNN only needs that file to code generate.

* You link the input directory with all trained models (infinite amount of **.keras** or **.h5** files if you please) wanting to be code generated as well as the output directory to save the files. Additionally, the data type precision is optional, but only accepts float or double.

    * **dump_model/** is the default linked dump directory to place trained models.

* The generated header file and "predict" function will be named after the file name of the trained model that it was code generated from.

    * Example: ***my_model.h5***  will code generate the header file  ***my_model.hpp*** and the predict function template will become ***auto my_model(nn_inputs)***.

    * Along with that, a **test.cpp** file will also be copied into the desired directory. This file makes sure logic and accuracy of the trained neural net matches with results during training. 

* CodeJeNN is not perfect. Thus, numerous errors (i.e. try catch blocks) are used to help the user zero in on the code generation errors. From testing, most of the time, errors are due to the in-capabilites of CodeJeNN.

    * If such errors occurs, the user is very, and I mean very encouraged to add and update the logic and submit a pull request and merge. 

## CodeJeNN Model Architecture (ii)

* CodeJeNN suports CNNs and MLPs. CodeJeNN only supports single input and output sequential models or functional models, more information can be found in the **../tutorial/** directory. Single input and single output refers to a scalar, an array, a 2d matrix or 3d matrix. 

    * Sequential models or functional models refers to how the user can build an MLP or CNN using tensorflow keras. You can stack different types of layers on top whether they are Dense, Conv2D, Activations, Normalization layers and etc. 

* Unfortunately, at the time of writing, tensorflow keras is currently implementing API for GNNs.

* To truly understand how complex and deep these architectures can be code generated, the user is referred to the **../tutorial/** directory.

* Once the model is code generated, the header file structure is as follows (percentage denotes space taken in the header file):
    1. import std libraries **>1%**
    1. layer propagation functions (Dense(), UnitNormalization(), Conv2DTranspose, etc...) **~10%**
    1. function header template **>1%**
    1. arrays of layer parameters (weights, baises, strides, kernels, gamma, etc...) **~85%**
    1. functions calls to drive through the each layer of the model. **~5%**

## Aglorithm Complexity (iii)

* CodeJeNN generates c++ interpreted nueral nets for computaional physics application, which means these NN are used within big codes. CodeJeNN takes advantage of what c++ has to offer in terms of runtime calculation speed. 

* It is taboo to use vectorization or parallelization within the c++ NN because outside of that, the code is assumed to be already vectorized and parallelized thus no threads are left and implementation of such actions can result in slower inference and slower code overall.

* However, CodeJeNN tries to inline as much as possible by using templating, lambda functions, less cache spaced used, etc.

    * Constexpr static arrays are used as much as possible and variables is passsed by constant reference or by pointer as much as possible to reduce memory usage. 

    * For Loops can be optimized using variadic templating which inlines and unravels the code, increasing compile time and cache space used, but significantly decreases run time. CodeJeNN sticks to for loops because of how long and extensive these weight and biases arrays can get. 

    * The layer propagation functions such as Conv1D(), Conv2D(), Dense(), LayerNormalization(), are inlined as much as possible and memroy is allocated beforehand in these functions as much as possible.

    * Activation functions are not considered layer propagation functions and are defined and lambda functions to inline as much as possible. In the layer propagation function template headers, the activation function is defined as a template parameter thus the activation functions are passed by lambda and inlined as much as possible. 

* The user is also encouraged to optimize the code generated neural net as much as possible too. 

## Standardization/Normalization (iv)

CodeJeNN supports normalization/standardization of inputs and outputs. 

* The standardization/normalization values must be stored in a **.npy** file.

* The normalization/standardization file must be named the following depending on how you wish to normalize:

    * `input_max.npy` `input_min.npy` `output_max.npy` `output_min.npy`
    
    * `input_std.npy` `intput_mean.npy` `output_std.npy` `output_mean.npy`

## Testing Generated Model (v)

* As said before, a test.cpp file will also be copied into the output directory so the user can compare the c++ NN output with the python tensorflow keras output. 

* The user may edit the **test.cpp** source file to check for precision and accuracy of trained neural net. Down below is what needs to be change in **test.cpp**

    **Header File** 
    ```c++
    // change file name to desired header file
    #include "header_file.hpp"
    ```
    **Input**
    ```c++
    // change input to desired features
    std::array<Scalar, "number_of_input_features"> input = {"input(s)"};
    auto output = "function_name"<Scalar>(input);
    ```
1. Once done, compile code. Comments on the preferred method to compile and run **test.cpp** is at the bottom of the source file.

1. Verify that the output is correct!

## Try An Example Already In **dump_model/** (vi)
HOPEFULLY YOU READ ALL THIS, you can now try out the example in **dump_model/**. Just open a terminal/shell in the **src/** directory, MAKE SURE YOU HAVE INSTALLED ALL DEPENDICIES FROM **requirements.txt**, link the correct directorys in **generate.sh**, type `bash generate.sh` in the terminal/shell, and you are good to go!

There are of course much more examples in `../tutorials/`.