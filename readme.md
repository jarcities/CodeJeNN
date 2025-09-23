<!-- 
Distribution Statement A. Approved for public release, distribution is unlimited.
---
THIS SOURCE CODE IS UNDER THE CUSTODY AND ADMINISTRATION OF THE GOVERNMENT OF THE UNITED STATES OF AMERICA.
BY USING, MODIFYING, OR DISSEMINATING THIS SOURCE CODE, YOU ACCEPT THE TERMS AND CONDITIONS IN THE NRL OPEN LICENSE AGREEMENT.
USE, MODIFICATION, AND DISSEMINATION ARE PERMITTED ONLY IN ACCORDANCE WITH THE TERMS AND CONDITIONS OF THE NRL OPEN LICENSE AGREEMENT.
NO OTHER RIGHTS OR LICENSES ARE GRANTED. UNAUTHORIZED USE, SALE, CONVEYANCE, DISPOSITION, OR MODIFICATION OF THIS SOURCE CODE
MAY RESULT IN CIVIL PENALTIES AND/OR CRIMINAL PENALTIES UNDER 18 U.S.C. § 641.
-->

![CodeJeNN](logo.png)

<div align="center">

_San Diego State University, San Diego, CA_

_Labratories of Computational Physics and Fluid Dynamics, Naval Research Lab, Washington, DC_

__Distribution Statement A: Distribution Statement A. Approved for public release, distribution is unlimited.__
</div>

## Introduction

CodeJeNN is a simple neural network generator for c++ that can robustly ingest a trained NN to be used on the fly for inference in target computational physics and fluid dynamics software. This abstracts away the need for using third party libraries which are often cumbersome and would require shipping large data onto main memory to utilize inference. Instead, the NN is inlined and localized to the users machine for inference. This will assure scalability and allow for certain optimizations that ML has promised to numerical solvers: faster constiutive laws, accurate interpolation functions, etc. CodeJeNN works by converting a trained neural net stored in a .h5 or a .keras file into a c++ header file that can be used in the users code to predict (perform inference).

## File Structure
```plaintext
CodeJeNN/
    ├── README.md
    └── tutorials/
            └── simple_mlp_01
            └── 1d_cnn_02
            └── model_file_breakdown.md
            └── supported_layers.md
    └── src/
            └── bin/
            └── codegen/
            └── dump_model/
            └── testing/
            └── clean.sh
            └── generate.sh
            └── readme.md
    └── citation.cff
    └── license.txt
    └── logo.png
    └── readme.md
    └── requirements.txt
```

## Starting Point

* **Python 3.11 is required becuase the latest version of tensorflow requires it.**

* **You ONLY need the `src` directory, all other files are just auxillary but still very useful.**

1. You do not need to create a virtual environment, but it is best to use one. You can install python 3.11 in your home directory and `cd` to your home directory and use. 
    ```bash
    python3.11 -m venv codejenn
    source codejenn/bin/activate
    ```
    Or you can install conda and use:
    ```bash
    conda create -n codejenn python=3.11
    conda activate codejenn
    ```
    Where `codejenn` is the name of the environment.

1. Next install the necessary packages which are common in most deep learning codes already.
    ```bash
    pip install -r requirements.txt
    ```
1. From here, `cd` into `src` and carry on with the ***README.md*** file in there.