# Distribution Statement A. Approved for public release, distribution is unlimited.
"""
THIS SOURCE CODE IS UNDER THE CUSTODY AND ADMINISTRATION OF THE GOVERNMENT OF THE UNITED STATES OF AMERICA. BY USING, MODIFYING, OR DISSEMINATING THIS SOURCE CODE, YOU ACCEPT THE TERMS AND CONDITIONS IN THE NRL OPEN LICENSE AGREEMENT. USE, MODIFICATION, AND DISSEMINATION ARE PERMITTED ONLY IN ACCORDANCE WITH THE TERMS AND CONDITIONS OF THE NRL OPEN LICENSE AGREEMENT. NO OTHER RIGHTS OR LICENSES ARE GRANTED. UNAUTHORIZED USE, SALE, CONVEYANCE, DISPOSITION, OR MODIFICATION OF THIS SOURCE CODE MAY RESULT IN CIVIL PENALTIES AND/OR CRIMINAL PENALTIES UNDER 18 U.S.C. § 641.
"""

import os
import absl.logging
import warnings
absl.logging.set_verbosity('error')
warnings.filterwarnings("ignore", category=UserWarning, module='keras')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def testSource(precision_type):
    
    source_code = f"""
#include <iostream>
#include <array>
#include <random>
#include <cmath>
#include "header_file.h" // change file name to desired header file

using Scalar = {precision_type};

int main() {{
    std::array<Scalar, _number_of_input_features_> input = {{_inputs_}}; // change input to desired features
    auto output = _namespace_::_function_name_<Scalar>(input); // change input to desired features
    std::cout << "Output: ";
    for(const auto& val : output) {{
        std::cout << val << " ";
    }}
    std::cout << std::endl;
    return 0;
}}

/*
clang++ -std=c++2b -o test test.cpp
./test
*/
"""

    return source_code
