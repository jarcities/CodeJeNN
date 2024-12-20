<!-- Distribution Statement A. Approved for public release, distribution is unlimited.
...
THIS SOURCE CODE IS UNDER THE CUSTODY AND ADMINISTRATION OF THE GOVERNMENT OF THE UNITED STATES OF AMERICA. BY USING, MODIFYING, OR DISSEMINATING THIS SOURCE CODE, YOU ACCEPT THE TERMS AND CONDITIONS IN THE NRL OPEN LICENSE AGREEMENT. USE, MODIFICATION, AND DISSEMINATION ARE PERMITTED ONLY IN ACCORDANCE WITH THE TERMS AND CONDITIONS OF THE NRL OPEN LICENSE AGREEMENT. NO OTHER RIGHTS OR LICENSES ARE GRANTED. UNAUTHORIZED USE, SALE, CONVEYANCE, DISPOSITION, OR MODIFICATION OF THIS SOURCE CODE MAY RESULT IN CIVIL PENALTIES AND/OR CRIMINAL PENALTIES UNDER 18 U.S.C. § 641.
-->

## Directory Contents
  * **codegen/** ⮕ Folder that holds all methods/function to extract, load, and code generate any model in a directory where your neural nets are stored if they are not stored in **dump_model/**.
  * **main.py** ⮕ Main python script that runs source code for code generating any trained model into a C++ inference (predict) function. 
  * **README.md** ⮕ What you are currently reading you goober.
  * **generate.sh** ⮕ Bash script you can use if you want, or include the contents in your own bash script or makefile.
  * **dump_model/** ⮕ Dump folder that you can dump any trained model and link during the code generation process if you wish to not use a seperate directory.
  * **requirements.txt** are all the necessary python package/library dependicies.

## Code Generation Explanation
* CodeJeNN works by reading in a **.keras**,a **.h5**, or a **.onnx**. file that has a trained neural net and codegenerate that into a c++ header file. 
* CodeJeNN can code generate as many trained neural nets as you please. 
* You link the input folder with all trained models wanting to be code generated as well as the output folder to save the files (more information below). Additionally, the precision is optional, but only accepts float or double.
    * **dump_model/** is the default folder to place trained models.
* The generated header file and "predict" function will be named after the file name of the trained model that it was code generated from.
    * Example: `my_model.h5`  ⮕  `my_model.h` & `auto my_model(inputs)`
    * Along with that, a **test.cpp** will also be copied into the desired directory. To make sure logic and accuracy of the trained neural net matches with results during training. 
* CodeJeNN supports normalization and standardization of inputs and outputs. 
    * The standardization/normalization values must be stored in a **.txt**, **.dat**, or a **.csv** file with **SPACES AS DELIMITERS**.
    * The normalization/standardization file name must match the name of the **.keras**, **.h5**, or **.onnx** file to make sure the proper parameters are generated correctly.
    * A full example is shown below and in the **example.dat** file in **dump_model/**. The important thing to remember is to use space as delimiters, use the correct variable name, and put all array of values in brackets.
* All code generation uses static arrays for memory costs.

## Code Generation Steps
1. First create trained neural nets using keras, onnx, or tensorflow and save in the supported file extensions.
    * If necessary, save normalization/standardization parameters in a **.dat**, a **.csv**, or a **.txt** file format. 
1. Install necessary python libraries using `pip install > requirements.txt` in terminal/shell or whatever package manager you prefer.
1. Put all models wanting to be code generated in the **dump_model/** folder, or whatever directory you wish to link.
1. Link necessary directories or change the bash script **(generate.sh)** dependicies as stated below. 
    1. `--input` ⮕ path to folder that holds any and all trained models to be code generated, or use **dump_model/**.
    1. `--output` ⮕ path to folder to save all generated header files.
    1. `--precision` ⮕ (OPTIONAL) variable type of precision, either double or float. If not specified, will default to float.

    ```bash
    python main.py --input="path_to_input_folder" --output="path_to_output_folder" --precision="desired_precision"
    ```
1. Run **generate.sh** (type `bash generate.sh` in terminal/shell).
1. Once generation is complete. You are done!

## Testing Generated Header (Predict) Files
1. If you want, you must edit the **test.cpp** source file to check for precision and accuracy of trained neural net. Down below is what you have to change in **test.cpp**

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

1. Once done, compile code. Comments on the preferred method to compile and run **test.cpp** is at the bottom of the source file in comments
1. Verify that output is correct!


---

---

HOPEFULLY YOU READ ALL THIS, you can now try out the example in the **dump_model/** folder. Just open a terminal/shell in the **src/** directory, KEEP ONE OF THE OPTIONS FOR NORMALIZATION/STANDARDIZATION AND DELETE THIS REST, link the correct folders in **generate.sh**, type `bash generate.sh` in the terminal/shell, and you are good to go!