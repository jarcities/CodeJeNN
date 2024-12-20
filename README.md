![CodeJeNN](others/logo.png/)

<div align="center">

_San Diego State University, San Diego, CA_

_Labratories of Computational Physics and Fluid Dynamics, Naval Research Lab, Washington, DC_

__Distribution Statement A: Distribution Statement A. Approved for public release, distribution is unlimited.__
</div>

## Introduction

CodeJeNN is an interface package that can robustly ingest a trained NN to be used on the fly for inference in target computational fluid dynamics software. This abstracts away the need for using third party libraries which are often cumbersome and would require shipping CFD data onto main memory to utilize inference. Instead, we imbed the NN for inference onto the device itself. Currently the NNs are small enough and lean enough from PINN where they can be done directly on device. This will assure scalability and allow for certain optimizations that ML has promised to CFD: smaller look up tables and faster constiutive laws. CodeJeNN works by converting a trained neural net stored in a .onnx, .h5, or a .keras file into a c++ header file that can be used in the users code to predict (perform inference).

**You only need to download the `src` directory, all other files are just auxillary.**

## File Structure
```plaintext
CodeJeNN/
    ├── README.md
    └── examples/
            └── neural_net_builder/
    └── others/
            └── archive/      
                └── src_v1
                └── src_v2
                └── src_v3
                └── src_v4
            └── h5_file_breakdown.md
            └── layers.md
    └── src/
            └── codegen/
            └── dump_model/
            └── generate.sh
            └── README.md
    └── license.txt
```

## File Contents
* `examples/` : 
    * contains examples of trained neural nets in h5, onnx, keras file formats.
    * `neural_net_builder/` : python scripts to train neural nets and test the outputs.
* `others/` :
    * `archive/` : previous versions of codejenn.
    * `h5_file_breakdown.md` : explanation of how keras saves trained models in a hdf5 file.
    * `layers.md` : shows the layers that codejenn can codegenerates and supports for inference.
* `src/` : source code with its own **README** explaning how to run.
* `license.txt` : distribution A licensing. 
