#pragma once
#include <iostream>
#include <array>
#include <random>
#include <cmath>
#include <functional>

// Added for convolution and pooling functions
#include <limits>

//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\// 

template<typename Scalar, int size>
void layerNormalization(Scalar* outputs, const Scalar* inputs, const Scalar* gamma, const Scalar* beta, Scalar epsilon) noexcept {
    Scalar mean = 0;
    Scalar variance = 0;
    for (int i = 0; i < size; ++i) {
        mean += inputs[i];
    }
    mean /= size;
    for (int i = 0; i < size; ++i) {
        variance += (inputs[i] - mean) * (inputs[i] - mean);
    }
    variance /= size;
    for (int i = 0; i < size; ++i) {
        outputs[i] = gamma[i] * ((inputs[i] - mean) / std::sqrt(variance + epsilon)) + beta[i];
    }
}

template<typename Scalar, int size>
void batchNormalization(Scalar* outputs, const Scalar* inputs, const Scalar* gamma, const Scalar* beta, const Scalar* mean, const Scalar* variance, const Scalar epsilon) noexcept {
    for (int i = 0; i < size; ++i) {
        outputs[i] = gamma[i] * ((inputs[i] - mean[i]) / std::sqrt(variance[i] + epsilon)) + beta[i];
    }
}

template<typename Scalar, int output_size>
void forwardPass(Scalar* outputs, const Scalar* inputs, const Scalar* weights, const Scalar* biases, int input_size, activationFunction<Scalar> activation_function, Scalar alpha) noexcept {
    for(int i = 0; i < output_size; ++i){
        Scalar sum = 0;
        for(int j = 0; j < input_size; ++j){
            sum += inputs[j] * weights[j * output_size + i];
        }
        sum += biases[i];
        activation_function(outputs[i], sum, alpha);
    }
}

// --- Convolution Functions ---

// 2D Convolution with VALID padding
template<typename Scalar, int H_in, int W_in, int C_in, int H_k, int W_k, int C_out, int StrideH, int StrideW>
void conv2D_valid(const Scalar* input, const Scalar* kernel, const Scalar* bias, Scalar* output) noexcept {
    constexpr int H_out = (H_in - H_k) / StrideH + 1;
    constexpr int W_out = (W_in - W_k) / StrideW + 1;
    for (int h = 0; h < H_out; ++h) {
        for (int w = 0; w < W_out; ++w) {
            for (int c = 0; c < C_out; ++c) {
                Scalar sum = 0;
                for (int kh = 0; kh < H_k; ++kh) {
                    for (int kw = 0; kw < W_k; ++kw) {
                        for (int cin = 0; cin < C_in; ++cin) {
                            int kernel_index = ((kh * W_k + kw) * C_in + cin) * C_out + c;
                            int input_index = ((h * StrideH + kh) * W_in + (w * StrideW + kw)) * C_in + cin;
                            sum += input[input_index] * kernel[kernel_index];
                        }
                    }
                }
                output[(h * W_out + w) * C_out + c] = sum + bias[c];
            }
        }
    }
}

// 2D Convolution with SAME padding
template<typename Scalar, int H_in, int W_in, int C_in, int H_k, int W_k, int C_out, int StrideH, int StrideW>
void conv2D_same(const Scalar* input, const Scalar* kernel, const Scalar* bias, Scalar* output) noexcept {
    constexpr int pad_h = ((H_in - 1) * StrideH + H_k - H_in) / 2;
    constexpr int pad_w = ((W_in - 1) * StrideW + W_k - W_in) / 2;
    constexpr int H_out = (H_in + 2 * pad_h - H_k) / StrideH + 1;
    constexpr int W_out = (W_in + 2 * pad_w - W_k) / StrideW + 1;
    Scalar padded_input[(H_in + 2 * pad_h) * (W_in + 2 * pad_w) * C_in] = {0};
    for (int h = 0; h < H_in; ++h) {
        for (int w = 0; w < W_in; ++w) {
            for (int c = 0; c < C_in; ++c) {
                int in_index = (h * W_in + w) * C_in + c;
                int pad_index = ((h + pad_h) * (W_in + 2 * pad_w) + (w + pad_w)) * C_in + c;
                padded_input[pad_index] = input[in_index];
            }
        }
    }
    for (int h = 0; h < H_out; ++h) {
        for (int w = 0; w < W_out; ++w) {
            for (int c = 0; c < C_out; ++c) {
                Scalar sum = 0;
                for (int kh = 0; kh < H_k; ++kh) {
                    for (int kw = 0; kw < W_k; ++kw) {
                        for (int cin = 0; cin < C_in; ++cin) {
                            int kernel_index = ((kh * W_k + kw) * C_in + cin) * C_out + c;
                            int pad_index = ((h * StrideH + kh) * (W_in + 2 * pad_w) + (w * StrideW + kw)) * C_in + cin;
                            sum += padded_input[pad_index] * kernel[kernel_index];
                        }
                    }
                }
                output[(h * W_out + w) * C_out + c] = sum + bias[c];
            }
        }
    }
}

// Depthwise 2D Convolution with VALID padding
template<typename Scalar, int H_in, int W_in, int C_in, int H_k, int W_k, int depth_multiplier, int StrideH, int StrideW>
void depthwiseConv2D_valid(const Scalar* input, const Scalar* kernel, const Scalar* bias, Scalar* output) noexcept {
    constexpr int C_out = C_in * depth_multiplier;
    constexpr int H_out = (H_in - H_k) / StrideH + 1;
    constexpr int W_out = (W_in - W_k) / StrideW + 1;
    for (int h = 0; h < H_out; ++h) {
        for (int w = 0; w < W_out; ++w) {
            for (int cin = 0; cin < C_in; ++cin) {
                for (int m = 0; m < depth_multiplier; ++m) {
                    Scalar sum = 0;
                    for (int kh = 0; kh < H_k; ++kh) {
                        for (int kw = 0; kw < W_k; ++kw) {
                            int kernel_index = ((kh * W_k + kw) * C_in + cin) * depth_multiplier + m;
                            int input_index = ((h * StrideH + kh) * W_in + (w * StrideW + kw)) * C_in + cin;
                            sum += input[input_index] * kernel[kernel_index];
                        }
                    }
                    output[((h * W_out + w) * C_in + cin) * depth_multiplier + m] = sum + bias[cin * depth_multiplier + m];
                }
            }
        }
    }
}

// Depthwise 2D Convolution with SAME padding
template<typename Scalar, int H_in, int W_in, int C_in, int H_k, int W_k, int depth_multiplier, int StrideH, int StrideW>
void depthwiseConv2D_same(const Scalar* input, const Scalar* kernel, const Scalar* bias, Scalar* output) noexcept {
    constexpr int pad_h = ((H_in - 1) * StrideH + H_k - H_in) / 2;
    constexpr int pad_w = ((W_in - 1) * StrideW + W_k - W_in) / 2;
    constexpr int H_out = (H_in + 2 * pad_h - H_k) / StrideH + 1;
    constexpr int W_out = (W_in + 2 * pad_w - W_k) / StrideW + 1;
    Scalar padded_input[(H_in + 2 * pad_h) * (W_in + 2 * pad_w) * C_in] = {0};
    for (int h = 0; h < H_in; ++h) {
        for (int w = 0; w < W_in; ++w) {
            for (int c = 0; c < C_in; ++c) {
                int in_index = (h * W_in + w) * C_in + c;
                int pad_index = ((h + pad_h) * (W_in + 2 * pad_w) + (w + pad_w)) * C_in + c;
                padded_input[pad_index] = input[in_index];
            }
        }
    }
    for (int h = 0; h < H_out; ++h) {
        for (int w = 0; w < W_out; ++w) {
            for (int cin = 0; cin < C_in; ++cin) {
                for (int m = 0; m < depth_multiplier; ++m) {
                    Scalar sum = 0;
                    for (int kh = 0; kh < H_k; ++kh) {
                        for (int kw = 0; kw < W_k; ++kw) {
                            int kernel_index = ((kh * W_k + kw) * C_in + cin) * depth_multiplier + m;
                            int pad_index = ((h * StrideH + kh) * (W_in + 2 * pad_w) + (w * StrideW + kw)) * C_in + cin;
                            sum += padded_input[pad_index] * kernel[kernel_index];
                        }
                    }
                    output[((h * W_out + w) * C_in + cin) * depth_multiplier + m] = sum + bias[cin * depth_multiplier + m];
                }
            }
        }
    }
}

// SeparableConv2D (VALID padding): performs depthwise then pointwise convolution
template<typename Scalar, int H_in, int W_in, int C_in, int H_k, int W_k, int depth_multiplier, int C_out, int StrideH, int StrideW>
void separableConv2D_valid(const Scalar* input, const Scalar* depthwise_kernel, const Scalar* pointwise_kernel, const Scalar* bias, Scalar* output) noexcept {
    constexpr int H_dw = (H_in - H_k) / StrideH + 1;
    constexpr int W_dw = (W_in - W_k) / StrideW + 1;
    constexpr int C_dw = C_in * depth_multiplier;
    Scalar depthwise_output[H_dw * W_dw * C_dw] = {0};
    depthwiseConv2D_valid<Scalar, H_in, W_in, C_in, H_k, W_k, depth_multiplier, StrideH, StrideW>(input, depthwise_kernel, nullptr, depthwise_output);
    // Pointwise convolution: kernel shape [1, 1, C_dw, C_out]
    for (int i = 0; i < H_dw * W_dw; ++i) {
        for (int c = 0; c < C_out; ++c) {
            Scalar sum = 0;
            for (int k = 0; k < C_dw; ++k) {
                sum += depthwise_output[i * C_dw + k] * pointwise_kernel[k * C_out + c];
            }
            output[i * C_out + c] = sum + bias[c];
        }
    }
}

// --- Pooling Functions ---

template<typename Scalar, int H_in, int W_in, int C>
void maxPooling2D(const Scalar* input, int pool_h, int pool_w, int stride_h, int stride_w, Scalar* output) noexcept {
    int H_out = (H_in - pool_h) / stride_h + 1;
    int W_out = (W_in - pool_w) / stride_w + 1;
    for (int h = 0; h < H_out; ++h) {
        for (int w = 0; w < W_out; ++w) {
            for (int c = 0; c < C; ++c) {
                Scalar max_val = -std::numeric_limits<Scalar>::infinity();
                for (int ph = 0; ph < pool_h; ++ph) {
                    for (int pw = 0; pw < pool_w; ++pw) {
                        int in_index = ((h * stride_h + ph) * W_in + (w * stride_w + pw)) * C + c;
                        if (input[in_index] > max_val)
                            max_val = input[in_index];
                    }
                }
                output[(h * W_out + w) * C + c] = max_val;
            }
        }
    }
}

template<typename Scalar, int H_in, int W_in, int C>
void averagePooling2D(const Scalar* input, int pool_h, int pool_w, int stride_h, int stride_w, Scalar* output) noexcept {
    int H_out = (H_in - pool_h) / stride_h + 1;
    int W_out = (W_in - pool_w) / stride_w + 1;
    for (int h = 0; h < H_out; ++h) {
        for (int w = 0; w < W_out; ++w) {
            for (int c = 0; c < C; ++c) {
                Scalar sum = 0;
                for (int ph = 0; ph < pool_h; ++ph) {
                    for (int pw = 0; pw < pool_w; ++pw) {
                        int in_index = ((h * stride_h + ph) * W_in + (w * stride_w + pw)) * C + c;
                        sum += input[in_index];
                    }
                }
                output[(h * W_out + w) * C + c] = sum / (pool_h * pool_w);
            }
        }
    }
}

template<typename Scalar, int H_in, int W_in, int C>
void globalAveragePooling2D(const Scalar* input, Scalar* output) noexcept {
    int size = H_in * W_in;
    for (int c = 0; c < C; ++c) {
        Scalar sum = 0;
        for (int i = 0; i < size; ++i) {
            sum += input[i * C + c];
        }
        output[c] = sum / size;
    }
}

template<typename Scalar, int H_in, int W_in, int C>
void globalMaxPooling2D(const Scalar* input, Scalar* output) noexcept {
    int size = H_in * W_in;
    for (int c = 0; c < C; ++c) {
        Scalar max_val = -std::numeric_limits<Scalar>::infinity();
        for (int i = 0; i < size; ++i) {
            if (input[i * C + c] > max_val)
                max_val = input[i * C + c];
        }
        output[c] = max_val;
    }
}

template <typename Scalar = double>
auto cnn2(const std::array<Scalar, 64>& initial_input) { 
    std::array<Scalar, 64> model_input = initial_input;

    if (model_input.size() != 64) { throw std::invalid_argument("Invalid input size. Expected size: 64"); }

    //\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\// 

    // Layer 1 (DepthwiseConv2D) shapes:
    constexpr std::array<int, 3> layer_1_input_shape = {8, 8, 1};
    constexpr std::array<int, 3> layer_1_output_shape = {8, 8, 1};
    constexpr std::array<int, 2> layer_1_strides = {1, 1};
    constexpr std::array<int, 2> layer_1_dilation_rate = {1, 1};
    constexpr std::array<int, 4> layer_1_kernel_shape = {3, 3, 1, 1};

    // Layer 2 (Conv2D) shapes:
    constexpr std::array<int, 3> layer_2_input_shape = {8, 8, 1};
    constexpr std::array<int, 3> layer_2_output_shape = {8, 8, 8};
    constexpr std::array<int, 2> layer_2_strides = {1, 1};
    constexpr std::array<int, 2> layer_2_dilation_rate = {1, 1};
    constexpr std::array<int, 4> layer_2_kernel_shape = {1, 1, 1, 8};

    constexpr std::array<Scalar, 1> gamma_2 = {1.003717065e+00};

    constexpr std::array<Scalar, 1> beta_2 = {4.168121377e-04};

    constexpr std::array<Scalar, 1> mean_2 = {5.972413719e-02};

    constexpr std::array<Scalar, 1> variance_2 = {8.717867732e-01};

    constexpr Scalar epsilon_2 = 1.000000000e-03;

    // Layer 3 (SeparableConv2D) shapes:
    constexpr std::array<int, 3> layer_3_input_shape = {8, 8, 8};
    constexpr std::array<int, 3> layer_3_output_shape = {8, 8, 16};
    constexpr std::array<int, 2> layer_3_strides = {1, 1};
    constexpr std::array<int, 2> layer_3_dilation_rate = {1, 1};
    constexpr std::array<int, 4> layer_3_depthwise_kernel_shape = {3, 3, 8, 1};
    constexpr std::array<int, 4> layer_3_pointwise_kernel_shape = {1, 1, 8, 16};

    // Layer 4 (SeparableConv2D) shapes:
    constexpr std::array<int, 3> layer_4_input_shape = {8, 8, 16};
    constexpr std::array<int, 3> layer_4_output_shape = {8, 8, 16};
    constexpr std::array<int, 2> layer_4_strides = {1, 1};
    constexpr std::array<int, 2> layer_4_dilation_rate = {1, 1};
    constexpr std::array<int, 4> layer_4_depthwise_kernel_shape = {3, 3, 16, 1};
    constexpr std::array<int, 4> layer_4_pointwise_kernel_shape = {1, 1, 16, 16};

    // Layer 5 (GlobalAveragePooling2D) shapes:
    constexpr std::array<int, 3> layer_5_input_shape = {8, 8, 16};
    constexpr std::array<int, 1> layer_5_output_shape = {16};

    constexpr std::array<Scalar, 8> gamma_5 = {9.933949709e-01, 1.003688455e+00, 1.000033498e+00, 1.001585841e+00, 9.962819219e-01, 1.003177285e+00, 1.007835746e+00, 9.962002039e-01};

    constexpr std::array<Scalar, 8> beta_5 = {-7.676446345e-03, 1.702864887e-03, -5.697892047e-04, -1.011130749e-03, -9.419163689e-03, 3.146166215e-03, 9.072840214e-03, -5.643260665e-03};

    constexpr std::array<Scalar, 8> mean_5 = {-2.591326274e-02, 3.080378473e-02, 1.869287342e-02, -1.981435344e-02, 1.026751846e-02, 6.722514052e-03, -7.810470648e-03, -4.235666990e-02};

    constexpr std::array<Scalar, 8> variance_5 = {8.694076538e-01, 8.732753396e-01, 8.649318814e-01, 8.655180335e-01, 8.615244031e-01, 8.606932759e-01, 8.609102368e-01, 8.850330114e-01};

    constexpr Scalar epsilon_5 = 1.000000000e-03;

    //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// 

    auto linear = [](Scalar& output, Scalar input, Scalar alpha) noexcept {
        output = input;
    };

    //\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\// 

    std::array<Scalar, 64> layer_1_output;
    depthwiseConv2D_same<Scalar, 8, 8, 1, 3, 3, 1, 1, 1>( model_input.data(), weights_1.data(), biases_1.data(), layer_1_output.data() );

    std::array<Scalar, 512> layer_2_output;
    conv2D_same<Scalar, 8, 8, 1, 1, 1, 8, 1, 1>( layer_1_output.data(), weights_2.data(), biases_2.data(), layer_2_output.data() );

    std::array<Scalar, 1024> layer_3_output;
    separableConv2D_valid<Scalar, 8, 8, 8, 3, 3, 1, 16, 1, 1>( layer_2_output.data(), weights_3_depthwise.data(), weights_3_pointwise.data(), biases_3.data(), layer_3_output.data() );

    std::array<Scalar, 1024> layer_4_output;
    separableConv2D_valid<Scalar, 8, 8, 16, 3, 3, 1, 16, 1, 1>( layer_3_output.data(), weights_4_depthwise.data(), weights_4_pointwise.data(), biases_4.data(), layer_4_output.data() );

    std::array<Scalar, 16> layer_5_output;
    globalAveragePooling2D<Scalar, 8, 8, 16>( layer_4_output.data(), layer_5_output.data() );

    std::array<Scalar, 16> model_output = layer_5_output;

    return model_output;
}
