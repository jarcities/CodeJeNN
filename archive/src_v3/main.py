# Distribution Statement A. Approved for public release, distribution is unlimited.
import os
import shutil
import argparse
import numpy as np
from codegen import loadModel, extractModel, codeGen, actFunctions, testSource, preambleHeader

# arg parser to get input and output folder
# arg parser to get input and output folder
parser = argparse.ArgumentParser(description="code generate trained neural net files into a given directory.")
parser.add_argument('--input', type=str, required=True, help='path of folder with trained model files')
parser.add_argument('--output', type=str, required=True, help='path of folder to save generated header files')
parser.add_argument('--precision', type=str, required=False, help='precision type to run neural net, either "double" or "float"')
parser.add_argument('--header', type=str, required=False, help='you may rename the predict function e.g. "calc_mu"')
args = parser.parse_args()

# if precision exists use that, else default to float
if args.precision is not None:
    precision_type = args.precision
else:
    precision_type = "float"
if args.header is not None:
    header_name = args.header
else: 
    header_name = "predict"
model_dir = args.input
save_dir = args.output

# make sure make sure the folder with model file exists
if not os.path.exists(model_dir):
    print(f"\nFALSE!!!, '{model_dir}' AIN'T EXIST!!!\n")

# make sure make sure the destination folder exists
elif not os.path.exists(save_dir):
    print(f"\nFALSE!!!, '{save_dir}' AIN'T EXIST!!!\n")

else:
    # copy test source file to make sure header file predict function works
    source_code = testSource(precision_type)
    save_path = os.path.join(save_dir, "test")
    with open(f"{save_path}.cpp", "w") as f:
        f.write(source_code)
    print(save_path)

    # iterate through each file in model directory
    for file_name in os.listdir(model_dir):
        file_path = os.path.join(model_dir, file_name)

        # skip hidden files
        if file_name == '.gitkeep' or file_name.startswith('.'):
            continue

        # codegen each model file in model_dump
        if os.path.isfile(file_path):
            try:
                model, file_extension = loadModel(file_path)
                weights_list, biases_list, activation_functions, alphas, dropout_rates, batch_norm_params, input_size = extractModel(model, file_extension)
                base_file_name = os.path.splitext(file_name)[0]
                save_path = os.path.join(save_dir, base_file_name)
                cpp_code = preambleHeader()
                cpp_code = actFunctions(cpp_code, activation_functions)
                cpp_code = codeGen(cpp_code, weights_list, biases_list, activation_functions, alphas, dropout_rates, batch_norm_params, input_size, save_path, header_name)
                with open(f"{save_path}.h", "w") as f:
                    f.write(cpp_code)
                print(save_path)
            except ValueError as e:
                print(f"\n - -  SOMETHING IS WRONG WITH --> '{file_name}': {e}  - -\n")
