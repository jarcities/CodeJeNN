[comment]: <> (Distribution Statement A: Distribution Statement A. Approved for public release, distribution is unlimited.)

## File Content
  * **codegen.py** ⮕ Script that holds all methods to extract, load, and codegen any model in dump folder. Includes all neural net predict methods.
  * **main.py** ⮕ Main python script that runs code generation of any model into a C++ inference (predict) function. 
  * **README.md** ⮕ What you are currently reading you goober.
  * **generate.sh** ⮕ Bash script user can use if they want. 
  * **dump_model** ⮕ Dump folder that user can dump any trained model and link that if they want. 

## Compilation Requirements
**Libraries:**
```zsh
pip install onnx onnxruntime tf2onnx torch tensorflow keras2onnx onnx2keras h5py numpy scipy keras scikit-learn absl-py
```

**Compiler Check:**
```zsh
/usr/bin/clang++ --version
/usr/bin/lldb --version
```

**Compiler and C++ Version used to write and test code**:
* C++14 up to C++23
* Homebrew clang version 18.1.8
* Target: arm64-apple-darwin23.5.0


## Compilation Notes

* The code is a single line terminal command script. 
* You linke the input folder with all trained models wanting to be code generated as well as the output folder to save the files. Additionally, the precision is optional, but only accepts float or double.
* All file name and namespaces of the generated header files are the same name of the model file that was in the input folder with the exceptions of hyphens and whitespace.
* Along with that, a **test.cpp** will also be copied into the desired directory. To make sure logic and accuracy of the trained neural net matches with results during training. 
* The following formats are supported: **.keras**, **.h5**, **.onnx**. Tensorflows **SavedModel** format is a little eeehhhh (may or may not work, but lean towards not working).
* All code generation uses static arrays for memory costs. Additionally, you may use the **dump_model** directory and link that if easier to user.

## Compilation Steps
1. `--input` ⮕ path to folder that holds any and all trained models to be code generated. 

    OR ⮕ dump any and all trained models in **dump_model** and link that instead.
1. `--output` ⮕ path to folder to save all generated header files.
1. `--precision` ⮕ (OPTIONAL) variable type of precision, either double or float. If not specified, will default to float.
1. `--header` ⮕ (OPTIONAL) name of the predict function at your disgression.

**NOW**:

Assuming you are in `src_v3`, you can run the terminal command below and define all parameters:
```zsh
python3 main.py --input="path_to_input_folder" --output="path_to_output_folder" --precision="desired_precision" --header="desired_function_name"
```

**OR**:

Assuming you are in `src_v3`, you can run the bash script and change the parameters in there:
```zsh
./generate.sh
```

## Post Compilation Steps
The two requirements to call the generated `predict` function is defining:

**Header File** 
```c++
// change file name to desired header file
#include "header_file.h"
```

**Input**
```c++
// change input to desired features
std::array<Scalar, "number_of_input_features"> input = {"input(s)"};
auto output = "namespace"::"function_name"<Scalar>(input);
```
