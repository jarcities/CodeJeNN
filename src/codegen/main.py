# Section: Library Imports
import os
import shutil
import argparse
import numpy as np
from layer_propagation import layer_propagation
from code_generation import preambleHeader, codeGen
from extract_model import extractModel
from load_model import loadModel
from test_script import testSource
from normalization_parameters import normParam

# Section: Argument Parsing
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
parser.add_argument(
    "--precision",
    type=str,
    required=False,
    help='precision type to run neural net, either "double" or "float"',
)
args = parser.parse_args()

# Section: Validate Precision Input
if args.precision is not None:
    if args.precision not in ["float", "double"]:
        print("\nERROR: Precision type must be 'float' or 'double'.\n")
        exit(1)
    precision_type = args.precision
else:
    precision_type = "float"

model_dir = args.input
save_dir = args.output

# Section: Check Directory Existence and Setup
if not os.path.exists(model_dir):
    print(f"ERROR: Input directory '{model_dir}' does not exist.")
    exit(1)
elif not os.path.exists(save_dir):
    print(f"WARNING: Output directory '{save_dir}' does not exist. Creating it now...")
    os.makedirs(save_dir)
else:
    # Section: Process Each Model File
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

        if os.path.isfile(file_path):
            try:
                base_file_name = os.path.splitext(file_name)[0]

                # Section: Load Normalization Parameters (if available)
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
                    input_norms, input_mins, output_norms, output_mins = (None, None, None, None)

                # Section: Load Model File
                model, file_extension = loadModel(file_path)
                
                # Section: Extract Model Parameters
                (
                    weights_list,
                    biases_list,
                    activation_functions,
                    alphas,
                    dropout_rates,
                    batch_norm_params,
                    conv_layer_params,
                    input_size,
                    layer_shape,
                    layer_type,
                ) = extractModel(model, file_extension)
                print()
                print("activation functions")
                print(len(activation_functions))
                print(activation_functions)
                print()
                print("layer type")
                print(len(layer_type))
                print(layer_type)
                print()

                # Section: Generate C++ Code from Model
                base_file_name = os.path.splitext(file_name)[0]
                save_path = os.path.join(save_dir, base_file_name)
                cpp_code = preambleHeader()

                # Process activations and lambda definitions
                cpp_code, cpp_lambda = layer_propagation(
                    cpp_code, activation_functions, layer_type
                )

                # Generate final code, passing conv_layer_params among others.
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
                    input_size,
                    save_path,
                    input_norms,
                    input_mins,
                    output_norms,
                    output_mins,
                    layer_shape,
                    layer_type,
                )

                print()
                with open(f"{save_path}.h", "w") as f:
                    f.write(cpp_code)
                print(save_path)

            except ValueError as e:
                print(f"\n - -  file type is not readable --> '{file_name}': {e}  - -\n")

    print()
