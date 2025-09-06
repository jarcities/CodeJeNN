# Distribution Statement A. Approved for public release, distribution is unlimited.
# ---
# THIS SOURCE CODE IS UNDER THE CUSTODY AND ADMINISTRATION OF THE GOVERNMENT OF THE UNITED STATES OF AMERICA.
# BY USING, MODIFYING, OR DISSEMINATING THIS SOURCE CODE, YOU ACCEPT THE TERMS AND CONDITIONS IN THE NRL OPEN LICENSE AGREEMENT.
# USE, MODIFICATION, AND DISSEMINATION ARE PERMITTED ONLY IN ACCORDANCE WITH THE TERMS AND CONDITIONS OF THE NRL OPEN LICENSE AGREEMENT.
# NO OTHER RIGHTS OR LICENSES ARE GRANTED. UNAUTHORIZED USE, SALE, CONVEYANCE, DISPOSITION, OR MODIFICATION OF THIS SOURCE CODE
# MAY RESULT IN CIVIL PENALTIES AND/OR CRIMINAL PENALTIES UNDER 18 U.S.C. § 641.

#!/bin/bash

### example ###
# python main.py --input="path_to_input_folder" --output="path_to_output_folder" --precision="desired_precision" 

### default ###
python3 ./training/mlp_eigen_parallel.py
python3 \
    ./codegen/main.py \
    --input="./dump_model" \
    --output="./bin" \
    --precision="double" \
    --custom_activation="nonzero_diag"

rm -rf .vscode/ codegen/__pycache__ dump_model/__pycache__