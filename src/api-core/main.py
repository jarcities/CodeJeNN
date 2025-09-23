"""
Distribution Statement A. Approved for public release, distribution is unlimited.
---
THIS SOURCE CODE IS UNDER THE CUSTODY AND ADMINISTRATION OF THE GOVERNMENT OF THE UNITED STATES OF AMERICA.
BY USING, MODIFYING, OR DISSEMINATING THIS SOURCE CODE, YOU ACCEPT THE TERMS AND CONDITIONS IN THE NRL OPEN LICENSE AGREEMENT.
USE, MODIFICATION, AND DISSEMINATION ARE PERMITTED ONLY IN ACCORDANCE WITH THE TERMS AND CONDITIONS OF THE NRL OPEN LICENSE AGREEMENT.
NO OTHER RIGHTS OR LICENSES ARE GRANTED. UNAUTHORIZED USE, SALE, CONVEYANCE, DISPOSITION, OR MODIFICATION OF THIS SOURCE CODE
MAY RESULT IN CIVIL PENALTIES AND/OR CRIMINAL PENALTIES UNDER 18 U.S.C. § 641.
"""

from calendar import c
from encodings.punycode import T
import os
import argparse
import numpy as np
from tensorflow.keras.utils import plot_model
from load_model import loadModel
from extract_model import extractModel
from build_model import buildModel
from code_generate import preambleHeader, codeGen
from test_script import testSource
from normalization import normParam

## ARG PARSING ##
parser = argparse.ArgumentParser(
    description="code generate trained neural net files into a given directory."
)
parser.add_argument(
    "--input", type=str, required=True, help="path of folder with trained model files"
)
parser.add_argument(
    "--output",
    type=str,
    required=True,
    help="path of folder to save generated header files",
)
precision_group = parser.add_mutually_exclusive_group()
precision_group.add_argument(
    "--double",
    action="store_true",
    help="use double precision for neural net computations",
)
precision_group.add_argument(
    "--float",
    action="store_true",
    help="use float precision for neural net computations (default)",
)
parser.add_argument(
    "--custom_activation",
    type=str,
    required=False,
    help="custom activation function to use, if any",
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="include debug print statements in the generated code",
)
parser.add_argument(
    "--model_image",
    action="store_true",
    help="save a visualization of the model architecture as a PNG file",
)
# process arguments
args = parser.parse_args()

## DATA TYPE PRECISION ##
if args.double:
    precision_type = "double"
elif args.float:
    precision_type = "float"
else:
    precision_type = "float"  # default to float

model_dir = args.input
save_dir = args.output

if args.custom_activation is not None:
    user_activation = args.custom_activation
else:
    user_activation = None

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
        if file_name.endswith(".npy"):
            continue

        ## CHECK FILE AND PROCESS MODEL ##
        if os.path.isfile(file_path):
            try:
                base_file_name = os.path.splitext(file_name)[0]

                #########################################
                ## 1. PROCESS NORMALIZATION PARAMETERS ##
                #########################################
                input_scale, input_shift, output_scale, output_shift = normParam(
                    model_dir
                )

                ###################
                ## 2. LOAD MODEL ##
                ###################
                try:
                    model, file_extension = loadModel(
                        file_path, base_file_name, user_activation
                    )
                    # model.summary()
                except ValueError as e:
                    print(f"\n-> <{file_name}> is NOT a compatible file (skipping).")
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
                        norm_layer_params,
                        conv_layer_params,
                        input_flat_size,
                        output_flat_size,
                        layer_shape,
                        layer_type,
                    ) = extractModel(model, file_extension, base_file_name)

                    #https://keras.io/api/utils/model_plotting_utils/
                    if args.model_image:
                        plot_model(
                            model,
                            to_file=f"{base_file_name}_architecture.png",
                            show_shapes=True,
                            show_dtype=True,
                            show_layer_names=True,
                            show_layer_activations=True,
                        )

                    ## DEBUG PRINTS ##
                    if args.debug:
                        print(f"\nModel Summary for {file_name}:")
                        print("----------------------------------")
                        model.summary()
                        print(f"\nWhat CodeJeNN extracted for {file_name}:")
                        print("------------------------------------------------------")
                        print(f"Input Size -> {input_flat_size}\n")
                        print(f"Output Size -> {output_flat_size}\n")
                        print(f"Layer Shape [{len(layer_shape)}] -> {layer_shape}\n")
                        print(f"Layer Types [{len(layer_type)}] -> {layer_type}\n")
                        print(
                            f"Activation Functions [{len(activation_functions)}] -> {activation_functions}\n"
                        )
                        print(
                            f"Convolutional Params (# of params per layer) [{len(conv_layer_params)}] -> [",
                            end="",
                        )
                        for i, params in enumerate(conv_layer_params):
                            if params is not None:
                                num_of_info = len(params)
                                print(f"{num_of_info}", end="")
                            else:
                                print("-", end="")
                            if i < len(conv_layer_params) - 1:
                                print(",", end=" ")
                        print("]", end="")
                        print("\n")
                        print(
                            f"Normalization Params (# of params per layer) [{len(norm_layer_params)}] -> [",
                            end="",
                        )
                        for i, params in enumerate(norm_layer_params):
                            if params is not None:
                                num_of_info = len(params)
                                print(f"{num_of_info}", end="")
                            else:
                                print("-", end="")
                            if i < len(norm_layer_params) - 1:
                                print(",", end=" ")
                        print("]", end="")
                        print(
                            "\n------------------------------------------------------"
                        )

                except ValueError as e:
                    print("\nError in extracting model:", e)
                    continue

                ############################
                ## 4. INITIALIZE C++ CODE ##
                ############################
                save_path = os.path.join(save_dir, base_file_name)
                cpp_code = preambleHeader()

                ######################
                ## 5. REBUILD MODEL ##
                ######################
                try:
                    cpp_code, cpp_lambda = buildModel(
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
                        norm_layer_params,
                        conv_layer_params,
                        input_flat_size,
                        output_flat_size,
                        save_path,
                        input_scale,
                        input_shift,
                        output_scale,
                        output_shift,
                        layer_shape,
                        layer_type,
                        base_file_name,
                        user_activation,
                        args.debug,
                    )
                except ValueError as e:
                    print("\nError in generating C++ code:", e)
                    continue

                print()
                with open(f"{save_path}.hpp", "w") as f:
                    f.write(cpp_code)
                print(f"Saved {file_name} model in {save_path}")

            except ValueError as e:
                print(f"\nERROR: '{file_name}' is not readable (skipping): {e}  - -\n")
                continue

print("\nAll done!\n")
