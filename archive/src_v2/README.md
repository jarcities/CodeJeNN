## File Content
  * **model_dump** ⮕ Dump folder for model files holding neural net architecture, parameter, data, etc.
  * **codegen_methods.py** ⮕ Script that holds all methods to extract, load, and codegen any model in dump folder.
  * **main.py** ⮕ Main python script that runs code generation of any model into a C++ inference function. 
  * **model_methods.h** ⮕ Header file with neural net methods definition and declaration.
  * **test.cpp** ⮕ Source file to test the predict function that will be code generated.
  * **README.md** ⮕ What you are currently reading you goober.

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

* The code (main script) works by dropping in any model you want into **dump_folder** to be generated into c++, which then become header files and saves it at the location path you gave when prompted.
* All file name and namespaces of the generated header files are the same name of the model file that was dumped with the exceptions of hyphens and whitespace.
* Along with that, **model_methods.h** and **test.cpp** will also be copied into the desired directory.
* The following formats are supported: **.keras**, **.h5**, **.onnx**. Tensorflows **SavedModel** format is a little eeehhhh (may or may not work, but lean towards not working).
* If memory allocation is important, everything should be an **std::array** except the input and output arrays which are **std::vector**.
* Code supporst only **float** and **double** as precision.

## Compilation Steps

1. Drop any model files in **model_dump**.
1. Run **main.py**.
1. Give the path of the directory you want to save all generated models when prompted.

## Post Compilation Steps
The three requirements to call the generated `predict` funciton is defining:

**Header File** 
```c++
// change file name to desired header file
#include "<header_file>.h"
```

**Precision**
```c++
// change Scalar to desired precision
using Scalar = "<precision>";
```

**Input**
```c++
// change input to desired features
std::vector<Scalar> input = {"<input array>"};
std::vector<Scalar> output = "<namespace>"::predict(input);
```