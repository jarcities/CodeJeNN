# Minimum CMake version requirement
cmake_minimum_required(VERSION 3.15)

# Project name and version
project(runtime_test VERSION 1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set compiler
set(CMAKE_CXX_COMPILER "clang++")

# Set compilation flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

# Include directories
if(DEFINED ENV{CONDA_PREFIX})
    include_directories("$ENV{CONDA_PREFIX}/Library/include")
    link_directories("$ENV{CONDA_PREFIX}/Library/lib")
else()
    message(FATAL_ERROR "CONDA_PREFIX is not set. Please set it to your Conda environment path.")
endif()

# Link libraries
set(LIBS tbb)

# Define executable targets
add_executable(original_runtime original_test.cpp)
target_link_libraries(original_runtime PRIVATE ${LIBS})

add_executable(newVariadic_runtime newVariadic_test.cpp)
target_link_libraries(newVariadic_runtime PRIVATE ${LIBS})

add_executable(newForLoop_runtime newForLoop_test.cpp)
target_link_libraries(newForLoop_runtime PRIVATE ${LIBS})

add_executable(total_runtime total_time.cpp)
target_link_libraries(total_runtime PRIVATE ${LIBS})

# Optional: Define a custom target to run executables
# Note: CMake is primarily for building; running multiple times is better handled via scripts.
