from calendar import c
import os
import argparse
import numpy as np
from A_load_model import loadModel
from B_extract_model import extractModel
from C_layer_propagation import layer_propagation
from D_code_generation import preambleHeader, codeGen
from Z_test_script import testSource
from Z_normalization_parameters import normParam

## ARG PARSING ##
parser = argparse.ArgumentParser(
    description="code generate trained neural net files into a given directory."
)
parser.add_argument(
    "--input", 
    type=str, 
    required=True, 
    help="path of folder with trained model files"
)
parser.add_argument(
    "--output",
    type=str,
    required=True,
    help="path of folder to save generated header files"
)
parser.add_argument(
    "--precision",
    type=str,
    required=False,
    help='precision type to run neural net, either "double" or "float"'
)
parser.add_argument(
    "--custom_activation",
    type=str,
    required=False,
    help='custom activation function to use, if any'
)
args = parser.parse_args()

## DATA TYPE PRECISION ##
if args.precision is not None:
    if args.precision not in ["float", "double"]:
        print("\nERROR: Precision type must be 'float' or 'double'.\n")
        exit(1)
    precision_type = args.precision
else:
    precision_type = "float"

model_dir = args.input
save_dir = args.output

#############################################
if args.custom_activation is not None:
    user_activation = args.custom_activation
else:
    user_activation = None
# print(user_activation)
#############################################

## CHECK INPUT AND OUTPUT DIRECTORIES ##
if not os.path.exists(model_dir):
    print(f"ERROR: Input directory '{model_dir}' does not exist.")
    exit(1)
elif not os.path.exists(save_dir):
    print(f"WARNING: Output directory '{save_dir}' does not exist. Creating it now...")
    os.makedirs(save_dir)
else:

    ## PROCESS EACH MODEL IN INPUT DIRECTORY ##
    for file_name in os.listdir(model_dir):
        file_path = os.path.join(model_dir, file_name)

        if file_name == ".gitkeep" or file_name.startswith("."):
            continue
        if (
            file_name.endswith(".dat")
            or file_name.endswith(".csv")
            or file_name.endswith(".txt")
        ):
            continue

        ## CHECK FILE AND PROCESS MODEL ##
        if os.path.isfile(file_path):
            try:
                base_file_name = os.path.splitext(file_name)[0]

                #########################################
                ## 1. PROCESS NORMALIZATION PARAMETERS ##
                #########################################
                dat_file = os.path.join(model_dir, f"{base_file_name}.dat")
                csv_file = os.path.join(model_dir, f"{base_file_name}.csv")
                txt_file = os.path.join(model_dir, f"{base_file_name}.txt")

                if os.path.exists(dat_file):
                    input_norms, input_mins, output_norms, output_mins = normParam(
                        dat_file
                    )
                elif os.path.exists(csv_file):
                    input_norms, input_mins, output_norms, output_mins = normParam(
                        csv_file
                    )
                elif os.path.exists(txt_file):
                    input_norms, input_mins, output_norms, output_mins = normParam(
                        txt_file
                    )
                else:
                    input_norms, input_mins, output_norms, output_mins = (
                        None,
                        None,
                        None,
                        None,
                    )

                ###################
                ## 2. LOAD MODEL ##
                ###################
                try:
                    model, file_extension = loadModel(file_path, base_file_name, user_activation)
                    # model.summary()
                except ValueError as e:
                    print("\nError in loading model:", e)
                    continue

                #################################
                ## 3. EXTRACT MODEL EVERYTHING ##
                #################################
                try:
                    (
                        weights_list,
                        biases_list,
                        activation_functions,
                        activation_configs, 
                        alphas,
                        dropout_rates,
                        batch_norm_params,
                        conv_layer_params,
                        input_flat_size,
                        output_flat_size,
                        layer_shape,
                        layer_type,
                    ) = extractModel(model, file_extension, base_file_name)
                    # print(layer_type)
                    # print(activation_functions)
                except ValueError as e:
                    print("\nError in extracting model:", e)
                    continue

                ############################
                ## 4. INITIALIZE C++ CODE ##
                ############################
                save_path = os.path.join(save_dir, base_file_name)
                cpp_code = preambleHeader()

                ############################################
                ## 5. PROCESS LAYER PROPAGATION FUNCTIONS ##
                ############################################
                try:
                    cpp_code, cpp_lambda = layer_propagation(
                        cpp_code, activation_functions, layer_type, base_file_name
                    )
                except ValueError as e:
                    print("\nError in generating layer propagation functions:", e)
                    continue

                ################################
                ## 6. GENERATE FINAL C++ CODE ##
                ################################
                try:
                    cpp_code = codeGen(
                        cpp_code,
                        cpp_lambda,
                        precision_type,
                        weights_list,
                        biases_list,
                        activation_functions,
                        alphas,
                        dropout_rates,
                        batch_norm_params,
                        conv_layer_params,
                        input_flat_size,
                        output_flat_size,
                        save_path,
                        input_norms,
                        input_mins,
                        output_norms,
                        output_mins,
                        layer_shape,
                        layer_type,
                        base_file_name,
                        user_activation
                    )
                except ValueError as e:
                    print("\nError in generating C++ code:", e)
                    continue

                print()
                with open(f"{save_path}.hpp", "w") as f:
                    f.write(cpp_code)
                print("Saved model in ", save_path)

            except ValueError as e:
                print(f"\nERROR: '{file_name}' is not readable (skipping): {e}  - -\n")
                continue
