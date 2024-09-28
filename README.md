![CodeJeNN](others/logo/logoRyan.png/)

<div align="center">

_San Diego State University, San Diego, CA_

_Labratories of Computational Physics and Fluid Dynamics, Naval Research Lab, Washington, DC_
</div>

## Distribution Statement

Distribution Statement A: Distribution Statement A. Approved for public release, distribution is unlimited.

## Introduction
"CodeJenn is an interface package that can robustly ingest a trained NN to be used on the fly for inference in target computational fluid dynamics software. This abstracts away the need for using third party libraries which are often cumbersome and would require shipping CFD data onto main memory to utilize inference. Instead, we imbed the NN for inference onto the device itself. Currently the NNs are small enough and lean enough from PINN where they can be done directly on device. This will assure scalability and allow for certain optimizations that ML has promised to CFD: smaller look up tables and faster constiutive laws." 

-- Ryan F. Johnson, LCP-NRL

## File Structure
    CodeJeNN
    ├── README.md
    └── /archive
        └── src_v1
        └── src_v2
        └── src_v3
    └── /examples
        └── /nn_builder
            └── 'scripts to build models'
        └── 'saved trained models'
    └── /others
        └── /logo
        └── h5_file_breakdown.md
    └── /src
        └── /codegen
            └── 'source functions'
        └── /dump_model
        └── 'bash script'
        └── main.py
        └── README.md

