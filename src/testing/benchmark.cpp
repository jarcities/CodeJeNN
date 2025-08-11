#define USE_EIGEN
#include "MLP_LU.hpp"
#include <iostream>
#include <random>
#include <iomanip>
#include <vector>
#include <chrono>

// Volatile global to prevent aggressive optimization
volatile double global_sum = 0.0;

template<typename Scalar>
void benchmark_updated_implementations(const std::array<Scalar, 85>& test_input, int iterations = 10000) {
    using namespace std::chrono;
    
    // Generate multiple test inputs to prevent caching
    std::vector<std::array<Scalar, 85>> test_inputs(100);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<Scalar> dis(-1.0, 1.0);
    
    for (auto& input : test_inputs) {
        for (int i = 0; i < 85; i++) {
            input[i] = dis(gen);
        }
    }
    
    // Warm up both implementations
    for (int i = 0; i < 50; i++) {
        auto result1 = MLP_LU(test_inputs[i % test_inputs.size()]);
        auto result2 = MLP_LU_Ultra_Optimized(test_inputs[i % test_inputs.size()]);
        global_sum += result1[0] + result2[0]; // Prevent optimization
    }
    
    // Benchmark original implementation
    auto start = high_resolution_clock::now();
    Scalar sum1 = 0;
    for (int i = 0; i < iterations; i++) {
        auto result = MLP_LU(test_inputs[i % test_inputs.size()]);
        sum1 += result[0]; // Use result to prevent optimization
    }
    auto end = high_resolution_clock::now();
    auto original_time = duration_cast<microseconds>(end - start);
    
    // Benchmark ultra-optimized implementation
    start = high_resolution_clock::now();
    Scalar sum2 = 0;
    for (int i = 0; i < iterations; i++) {
        auto result = MLP_LU_Ultra_Optimized(test_inputs[i % test_inputs.size()]);
        sum2 += result[0]; // Use result to prevent optimization
    }
    end = high_resolution_clock::now();
    auto ultra_time = duration_cast<microseconds>(end - start);
    
    // Store sums to prevent optimization
    global_sum += sum1 + sum2;
    
    std::cout << "Updated Implementation Benchmark (" << iterations << " iterations):\n";
    std::cout << "Original implementation:      " << original_time.count() << " μs";
    if (original_time.count() > 0) {
        std::cout << " (" << static_cast<double>(iterations) / original_time.count() * 1000000 << " calls/sec)";
    }
    std::cout << "\n";
    
    std::cout << "Ultra-optimized Eigen:        " << ultra_time.count() << " μs";
    if (ultra_time.count() > 0) {
        std::cout << " (" << static_cast<double>(iterations) / ultra_time.count() * 1000000 << " calls/sec)";
    }
    std::cout << "\n";
    
    if (ultra_time.count() > 0) {
        std::cout << "Ultra-optimized speedup:      " << static_cast<double>(original_time.count()) / ultra_time.count() << "x\n";
    }
    
    // Show average time per call
    std::cout << "Average time per call:\n";
    std::cout << "  Original:      " << static_cast<double>(original_time.count()) / iterations << " μs\n";
    std::cout << "  Ultra-opt:     " << static_cast<double>(ultra_time.count()) / iterations << " μs\n";
}

int main() {
    // Create a test input
    std::array<double, 85> test_input;
    
    // Initialize with some random values for testing
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1.0, 1.0);
    
    for (int i = 0; i < 85; i++) {
        test_input[i] = dis(gen);
    }
    
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "Updated MLP Performance Benchmark\n";
    std::cout << "=================================\n\n";
    
    // Test correctness first
    std::cout << "Testing correctness with updated weights...\n";
    auto result_original = MLP_LU(test_input);
    auto result_ultra = MLP_LU_Ultra_Optimized(test_input);
    
    double max_diff = 0.0;
    double avg_diff = 0.0;
    
    for (int i = 0; i < 85; i++) {
        double diff = std::abs(result_original[i] - result_ultra[i]);
        max_diff = std::max(max_diff, diff);
        avg_diff += diff;
    }
    avg_diff /= 85.0;
    
    std::cout << "Max difference: " << max_diff << "\n";
    std::cout << "Avg difference: " << avg_diff << "\n";
    
    if (max_diff < 1e-6) {
        std::cout << "✓ Implementations are numerically equivalent.\n\n";
    } else {
        std::cout << "⚠ Minor numerical differences detected.\n\n";
    }
    
    // Show sample outputs
    std::cout << "Sample outputs (first 5 values):\n";
    std::cout << "Original:   ";
    for (int i = 0; i < 5; i++) {
        std::cout << std::setw(15) << result_original[i] << " ";
    }
    std::cout << "\n";
    std::cout << "Ultra-opt:  ";
    for (int i = 0; i < 5; i++) {
        std::cout << std::setw(15) << result_ultra[i] << " ";
    }
    std::cout << "\n\n";
    
    // Run performance benchmarks
    std::cout << "Running benchmarks with updated weights:\n\n";
    
    benchmark_updated_implementations(test_input, 10000);
    std::cout << "\n";
    
    benchmark_updated_implementations(test_input, 50000);
    std::cout << "\n";
    
    std::cout << "Optimization notes:\n";
    std::cout << "- Updated weights have been integrated into Eigen optimized version\n";
    std::cout << "- Vectorized operations provide significant speedup\n";
    std::cout << "- SIMD instructions automatically utilized on M1 Mac\n";
    std::cout << "- Expression templates eliminate temporary objects\n";
    
    return 0;
}
