import os
import shutil
from codegen_methods import loadModel, extractModel, codeGen

while True:
    save_dir = input("\nWHAT IS THY PATH OF THY FOLDER TO WHICH I SHALL SAVE ALL NEURALOGICAL NETWORKS TO: ").strip()
    if os.path.exists(save_dir):
        break
    print(f"\nI HAVE BEEN LIED TO, '{save_dir}' DOES NOT EXITS!!!.\n")

header_file = "model_methods.h"
shutil.copy(header_file, save_dir)
test_file = "test.cpp"
shutil.copy(test_file, save_dir)

model_dir = "model_dump"

if not os.path.exists(model_dir):
    print(f"\nDID YOU DELETE OR RENAME '{model_dir}', BECAUSE IT DOESN'T EXIST!!!\n")
else:
    for file_name in os.listdir(model_dir):
        file_path = os.path.join(model_dir, file_name)
        if file_name == '.gitkeep' or file_name.startswith('.'):
            continue
        if os.path.isfile(file_path):
            try:
                model, file_extension = loadModel(file_path)
                weights_list, biases_list, activation_functions, alphas, dropout_rates, input_size = extractModel(model, file_extension)
                base_file_name = os.path.splitext(file_name)[0]
                save_path = os.path.join(save_dir, base_file_name)
                cpp_code = codeGen(weights_list, biases_list, activation_functions, alphas, dropout_rates, input_size, save_path)
                with open(f"{save_path}.h", "w") as f:
                    f.write(cpp_code)
                print(save_path)
            except ValueError as e:
                print(f"\nTHIS FILE SUCKS --> '{file_name}': {e}, SO I AM SKIPPING IT!!!\n")
