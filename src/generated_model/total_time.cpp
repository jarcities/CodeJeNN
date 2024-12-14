#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

// Function to extract the numeric value from a line
bool extractValue(const std::string& line, int& value) {
    std::istringstream iss(line);
    std::string token;

    // Attempt to read the numeric part
    if (!(iss >> value)) {
        return false; // Failed to read an integer
    }

    // Optionally, you can verify that the next token is "ms"
    if (!(iss >> token)) {
        return false; // Failed to read the "ms" part
    }

    // Optionally, verify that the token is "ms"
    if (token != "ms") {
        return false; // Unexpected token
    }

    return true; // Successfully extracted the value
}

// Function to sum values from a given file
long long sumFile(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Unable to open file '" << filename << "'." << std::endl;
        return 0;
    }

    long long sum = 0; // Use long long to prevent potential overflow
    std::string line;
    int value;
    int lineNumber = 0;

    while (std::getline(infile, line)) {
        lineNumber++;
        if (extractValue(line, value)) {
            sum += value;
        } else {
            std::cerr << "Warning: Failed to parse line " << lineNumber << " in '" << filename << "': " << line << std::endl;
        }
    }

    infile.close();
    return sum;
}

int main() {
    // Define the filenames
    const std::string file1 = "original_runtime.txt";
    const std::string file2 = "newArray_runtime.txt";

    // Sum values from file1
    long long sum1 = sumFile(file1);
    std::cout << "Total sum of " << file1 << ": " << sum1 << " ms" << std::endl;

    // Sum values from file2
    long long sum2 = sumFile(file2);
    std::cout << "Total sum of " << file2 << ": " << sum2 << " ms" << std::endl;

    return 0;
}
