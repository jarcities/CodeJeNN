#!/bin/bash
rm -f *.h
rm -f *.cpp

### example ###
# python main.py --input="path_to_input_folder" --output="path_to_output_folder" --precision="desired_precision" 

### windows ###
# python main.py --input="..\src\dump_model" --output="..\src" --precision="double" 

### unix/linux ###
python main.py --input="../src/dump_model" --output="../src" --precision="double"
