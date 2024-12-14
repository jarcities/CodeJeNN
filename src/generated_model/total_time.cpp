#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

bool extractValue(const std::string& line, int& value) {
    std::istringstream iss(line);
    std::string token;
    if (!(iss >> value)) return false;
    if (!(iss >> token)) return false;
    if (token != "ms") return false;
    return true;
}

void processFile(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Unable to open file '" << filename << "'." << std::endl;
        return;
    }
    long long sum = 0;
    int count = 0, value;
    std::string line;
    int lineNumber = 0;
    while (std::getline(infile, line)) {
        lineNumber++;
        if (extractValue(line, value)) {
            sum += value;
            count++;
        } else {
            std::cerr << "Warning: Failed to parse line " << lineNumber << " in '" << filename << "': " << line << std::endl;
        }
    }
    infile.close();
    std::cout << "Total sum of " << filename << ": " << sum << " ms" << std::endl;
    if (count > 0) std::cout << "Average time in " << filename << ": " << (sum / count) << " ms" << std::endl;
}

int main() {
    const std::string file1 = "original_runtime.txt";
    const std::string file2 = "newArray_runtime.txt";
    const std::string file3 = "newForLoop_runtime.txt";
    processFile(file1);
    processFile(file2);
    processFile(file3);
    return 0;
}
