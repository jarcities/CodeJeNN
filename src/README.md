<!-- Distribution Statement A. Approved for public release, distribution is unlimited.
...
THIS SOURCE CODE IS UNDER THE CUSTODY AND ADMINISTRATION OF THE GOVERNMENT OF THE UNITED STATES OF AMERICA. BY USING, MODIFYING, OR DISSEMINATING THIS SOURCE CODE, YOU ACCEPT THE TERMS AND CONDITIONS IN THE NRL OPEN LICENSE AGREEMENT. USE, MODIFICATION, AND DISSEMINATION ARE PERMITTED ONLY IN ACCORDANCE WITH THE TERMS AND CONDITIONS OF THE NRL OPEN LICENSE AGREEMENT. NO OTHER RIGHTS OR LICENSES ARE GRANTED. UNAUTHORIZED USE, SALE, CONVEYANCE, DISPOSITION, OR MODIFICATION OF THIS SOURCE CODE MAY RESULT IN CIVIL PENALTIES AND/OR CRIMINAL PENALTIES UNDER 18 U.S.C. § 641.
-->

## File Content
  * **codegen** ⮕ Folder that holds all methods to extract, load, and codegen any model in dump folder. Includes all neural net predict methods.
  * **main.py** ⮕ Main python script that runs code generation of any model into a C++ inference (predict) function. 
  * **README.md** ⮕ What you are currently reading you goober.
  * **generate.sh** ⮕ Bash script user can use if they want. 
  * **dump_model** ⮕ Dump folder that user can dump any trained model and link that if they want. 
  * **requirements.txt** are all the necessary python packages to successfully code generate. 

## Code Generation Notes
* You link the input folder with all trained models wanting to be code generated as well as the output folder to save the files. Additionally, the precision is optional, but only accepts float or double.
    * **dump_model** is the default folder to place trained models.
* All file name and namespaces of the generated header files are the same name of the model file that was in the input folder with the exceptions of hyphens and whitespace.
* Along with that, a **test.cpp** will also be copied into the desired directory. To make sure logic and accuracy of the trained neural net matches with results during training. 
* The following formats are supported: **.keras**, **.h5**, **.onnx**.
* All code generation uses static arrays for memory costs. Additionally, you may use the **dump_model** directory and link that if easier to user.

## Compilation/Build/Generation Steps
1. First create trained neural nets using keras, onnx, or tensorflow and save in the supported file extensions.
1. Install necessary python libraries using `pip install > requirements.txt` or whatever package manager you prefer.
1. Put all models wanting to be code generated in the **dump_model** folder, or whatever folder you wish to link.
1. Change the bash script **(generate.sh)** dependicies as stated below. 
    1. `--input` ⮕ path to folder that holds any and all trained models to be code generated, or use **dump_model**.
    1. `--output` ⮕ path to folder to save all generated header files.
    1. `--precision` ⮕ (OPTIONAL) variable type of precision, either double or float. If not specified, will default to float.

    ```zsh
    python main.py --input="path_to_input_folder" --output="path_to_output_folder" --precision="desired_precision"
    ```
1. Run **generate.sh** (type `bash generate.sh` in terminal/command shell).
1. Once generation is complete. You are done!

## Testing Generated Header Files
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

1. Once done, compile code. Comments on the preferred method to compile and run **test.cpp** is at the bottom of the source file.
1. Verify that output is correct. 