"""
Distribution Statement A. Approved for public release, distribution is unlimited.
---
THIS SOURCE CODE IS UNDER THE CUSTODY AND ADMINISTRATION OF THE GOVERNMENT OF THE UNITED STATES OF AMERICA.
BY USING, MODIFYING, OR DISSEMINATING THIS SOURCE CODE, YOU ACCEPT THE TERMS AND CONDITIONS IN THE NRL OPEN LICENSE AGREEMENT.
USE, MODIFICATION, AND DISSEMINATION ARE PERMITTED ONLY IN ACCORDANCE WITH THE TERMS AND CONDITIONS OF THE NRL OPEN LICENSE AGREEMENT.
NO OTHER RIGHTS OR LICENSES ARE GRANTED. UNAUTHORIZED USE, SALE, CONVEYANCE, DISPOSITION, OR MODIFICATION OF THIS SOURCE CODE
MAY RESULT IN CIVIL PENALTIES AND/OR CRIMINAL PENALTIES UNDER 18 U.S.C. § 641.
"""
import os
import absl.logging
import warnings
absl.logging.set_verbosity('error')
warnings.filterwarnings("ignore", category=UserWarning, module='keras')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def cppTestCode(precision_type, base_file_name, layer_shape):

    input_code = "\n"
    input_shape = layer_shape[0]

    if len(input_shape) == 1: 
        n = input_shape[0]
        input_code += f"\tstd::array<Scalar, {n}> input;\n\n"
        input_code += (
            f"    for (int i = 0; i < {n}; ++i) {{\n"
            "        input[i] = static_cast<Scalar>(i);\n"
            "    }\n"
        )
    elif len(input_shape) == 2:  
        rows, cols = input_shape
        input_code += f"\tstd::array<std::array<Scalar, {cols}>, {rows}> input;\n\n"
        input_code += (
            "    int val = 0;\n"
            f"    for (int i = 0; i < {rows}; ++i) {{\n"
            f"        for (int j = 0; j < {cols}; ++j) {{\n"
            "            input[i][j] = static_cast<Scalar>(val);\n"
            "            ++val;\n"
            "        }\n"
            "    }\n"
        )
    elif len(input_shape) == 3: 
        depth, rows, cols = input_shape
        input_code += f"\tstd::array<std::array<std::array<Scalar, {cols}>, {rows}>, {depth}> input;\n\n"
        input_code += (
            "    int val = 0;\n"
            f"    for (int d = 0; d < {depth}; ++d) {{\n"
            f"        for (int i = 0; i < {rows}; ++i) {{\n"
            f"            for (int j = 0; j < {cols}; ++j) {{\n"
            "                input[d][i][j] = static_cast<Scalar>(val);\n"
            "                ++val;\n"
            "            }\n"
            "        }\n"
            "    }\n"
        )
    else:
        raise ValueError("Unsupported input shape")

    test_code = f"""#include <iostream>
#include <array>
#include <random>
#include <cmath>
#include <iomanip>
#include "{base_file_name}.hpp"

using Scalar = {precision_type};

int main() {{
    {input_code}
    auto output = {base_file_name}<Scalar>(input);

    std::cout << std::scientific << std::setprecision(15);  // scientific notation precision
    std::cout << "Output:\\n";  
    for(const auto& val : output) {{
        std::cout << val << '\\n';
    }}
    std::cout << std::endl;

    return 0;
}}

/*
clang++ -std=c++23 -Wall -O3 -march=native -o test test.cpp
./test
*/
"""

    return test_code
