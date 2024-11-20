# Distribution Statement A. Approved for public release, distribution is unlimited.
"""
THIS SOURCE CODE IS UNDER THE CUSTODY AND ADMINISTRATION OF THE GOVERNMENT OF THE UNITED STATES OF AMERICA. BY USING, MODIFYING, OR DISSEMINATING THIS SOURCE CODE, YOU ACCEPT THE TERMS AND CONDITIONS IN THE NRL OPEN LICENSE AGREEMENT. USE, MODIFICATION, AND DISSEMINATION ARE PERMITTED ONLY IN ACCORDANCE WITH THE TERMS AND CONDITIONS OF THE NRL OPEN LICENSE AGREEMENT. NO OTHER RIGHTS OR LICENSES ARE GRANTED. UNAUTHORIZED USE, SALE, CONVEYANCE, DISPOSITION, OR MODIFICATION OF THIS SOURCE CODE MAY RESULT IN CIVIL PENALTIES AND/OR CRIMINAL PENALTIES UNDER 18 U.S.C. § 641.
"""

import os
import shutil
import argparse
import numpy as np
from codegen.activation_functions import activationFunctions
from codegen.codegen import preambleHeader, codeGen
from codegen.extract_model import extractModel
from codegen.load_model import loadModel
from codegen.test_script import testSource
from codegen.normalization_parameters import normParam

# arg parser to get input and output folder
parser = argparse.ArgumentParser(description="code generate trained neural net files into a given directory.")
parser.add_argument('--input', type=str, required=True, help='path of folder with trained model files')
parser.add_argument('--output', type=str, required=True, help='path of folder to save generated header files')
parser.add_argument('--precision', type=str, required=False, help='precision type to run neural net, either "double" or "float"')
args = parser.parse_args()

# if precision exists use that, else default to float
if args.precision is not None:
    precision_type = args.precision
else:
    precision_type = "float"
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
    print()
    print(save_path)

    # iterate through each file in model directory
    for file_name in os.listdir(model_dir):
        file_path = os.path.join(model_dir, file_name)

        # skip hidden files, dat and csv files
        if file_name == '.gitkeep' or file_name.startswith('.'):
            continue
        if file_name.endswith('.dat') or file_name.endswith('.csv') or file_name.endswith('.txt'):
            continue

        # codegen each model file in model_dump
        if os.path.isfile(file_path):
            try:
                # get file name
                base_file_name = os.path.splitext(file_name)[0]
                
                # check for corresponding .dat or .csv file
                dat_file = os.path.join(model_dir, f"{base_file_name}.dat")
                csv_file = os.path.join(model_dir, f"{base_file_name}.csv")
                txt_file = os.path.join(model_dir, f"{base_file_name}.txt")
                if os.path.exists(dat_file):
                    input_norms, input_mins, output_norms, output_mins = normParam(dat_file)
                elif os.path.exists(csv_file):
                    input_norms, input_mins, output_norms, output_mins = normParam(csv_file)
                elif os.path.exists(txt_file):
                    input_norms, input_mins, output_norms, output_mins = normParam(txt_file)
                else:
                    input_norms, input_mins, output_norms, output_mins = None, None, None, None

                model, file_extension = loadModel(file_path)
                # model.summary()
                # print("\n - - LOADED MODEL - - \n")

                weights_list, biases_list, activation_functions, alphas, dropout_rates, batch_norm_params, conv_layer_params, input_size = extractModel(model, file_extension)
                # print("\n - - EXTRACTED MODEL - - \n")

                base_file_name = os.path.splitext(file_name)[0]
                save_path = os.path.join(save_dir, base_file_name)
                cpp_code = preambleHeader()
                # print("\n - - CREATED PREAMBLE - - \n")

                cpp_code = activationFunctions(cpp_code, activation_functions)
                # print("\n - - CREATED ACTIVATION FUNCTIONS - - \n")

                cpp_code = codeGen(cpp_code, precision_type, weights_list, biases_list, activation_functions, alphas, dropout_rates, batch_norm_params, conv_layer_params, input_size, save_path, input_norms, input_mins, output_norms, output_mins)
                # print("\n - - GENERATED MODEL - - \n")

                print()
                with open(f"{save_path}.h", "w") as f:
                    f.write(cpp_code)
                print(save_path)
            except ValueError as e:
                print(f"\n - -  SOMETHING IS WRONG WITH --> '{file_name}': {e}  - -\n")

    print()