#!/bin/bash

### example ###
# python main.py --input="path_to_input_folder" --output="path_to_output_folder" --precision="desired_precision" 

### default ###
python3 ./codegen/main.py --input="./dump_model" --output="./bin" --precision="double"
