#!/bin/bash

## example ##
# python3 main.py --input="path_to_input_folder" --output="path_to_output_folder" --precision="desired_precision" 
# ./main.py --input="path_to_input_folder" --output="path_to_output_folder" --precision="desired_precision" 

## windows ##
python main.py --input="..\src\dump_model" --output="..\src" --precision="double" 

## mac ##
# python3 main.py --input="/Users/kingjarred/Documents/CODEJENN/src/dump_model" --output="/Users/kingjarred/Documents/CODEJENN/src" --precision="double"
