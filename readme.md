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

__Distribution Statement A: Distribution Statement A. Approved for public release, distribution is unlimited.__
</div>

## Introduction

CodeJeNN is a simple neural network generator for c++ that can robustly ingest a trained NN to be used on the fly for inference in target computational physics and fluid dynamics software. This abstracts away the need for using third party libraries which are often cumbersome and would require shipping large data onto main memory to utilize inference. Instead, the NN is inlined and localized to the users machine for inference. This will assure scalability and allow for certain optimizations that ML has promised to numerical solvers: faster constiutive laws, accurate interpolation functions, etc. CodeJeNN works by converting a trained neural net stored in a .h5 or a .keras file into a c++ header file that can be used in the users code to predict (perform inference).

## Directory Contents
```plaintext
CodeJeNN/
    └── src/
            └── api-core/
            └── bin/
            └── dump_model/
            └── clean.sh
            └── generate.sh
            └── readme.md
    └── tutorials/
            └── 01_simple_mlp/
            └── 02_cnn_1d/
            └── 03_cnn_2d/
            └── 04_cnn_3d/
            └── 05_advanced_mlp/
            └── hdf5_file_breakdown.md
            └── supported_layers.md
    └── citation.cff
    └── license.txt
    └── logo.png
    └── readme.md
    └── requirements.txt
```

## Starting Point

⇨ **Python 3.11 is required due to compatibility with the latest version of tensorflow (NOTE: Keras is used through tensorflow so only installing tensorflow is necessary).**

⇨ **The recommended installer is `apt` for Linux users and `brew` for macOS users.**

⇨ **You ONLY need the `src` directory, all other files are just auxillary but still very useful.**

1. First open up a terminal/shell session and clone this repo into the home "`~/`" directory:
    ```bash
    git clone https://github.com/jarcities/CodeJeNN.git ~/codejenn
    cd ~/codejenn
    ```
    or where ever you choose:
    ```bash
    git clone https://github.com/jarcities/CodeJeNN.git
    cd codejenn
    ```

1. You do not need to create a virtual environment, but it is best to use one. This allows all dependent packages to be in one spot. 

    The first way is by using conda which you can install from [Install Miniconda (official site)](https://www.anaconda.com/docs/getting-started/miniconda/install). Then in your terminal/shell:

    ```bash
    conda create -n codejenn python=3.11
    conda activate codejenn
    ```

    OR

    The second way is to use a python environment by installing python 3.11 using `sudo apt install python3.11` or `brew install python@3.11`. Then in your home directory create a **python_environments** directory and create an environtment in there.

    ```bash
    mkdir ~/python_environments/
    cd ~/python_environments/
    python3.11 -m venv codejenn
    source codejenn/bin/activate
    ```

    Where `codejenn` is the name of the environment.
    

1. Next install the necessary libraries which are common in most deep learning codes already.
    ```bash
    pip install -r requirements.txt
    ```
1. From here, `cd` into `src` and carry on with the ***README.md*** file in there.

## Citation
You may site this repo using `citation.cff` or PREFERABLY with the paper below.
```
#citation
```