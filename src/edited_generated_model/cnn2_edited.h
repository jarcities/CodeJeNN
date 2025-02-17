#pragma once
#include <iostream>
#include <array>
#include <random>
#include <cmath>
#include <functional>
#include <stdexcept>

template<typename Scalar>
using activationFunction = void(*)(Scalar&, Scalar, Scalar);

// If you have any extra includes or definitions, put them here.


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

template<typename Scalar, int out_channels, int out_height, int out_width>
void conv2DForward(Scalar* outputs, const Scalar* inputs, const Scalar* weights, const Scalar* biases,
                   int in_channels, int in_height, int in_width,
                   int kernel_h, int kernel_w, int stride_h, int stride_w,
                   int pad_h, int pad_w,
                   activationFunction<Scalar> activation_function, Scalar alpha) noexcept {
    for (int oc = 0; oc < out_channels; ++oc) {
        for (int oh = 0; oh < out_height; ++oh) {
            for (int ow = 0; ow < out_width; ++ow) {
                Scalar sum = 0;
                for (int ic = 0; ic < in_channels; ++ic) {
                    for (int kh = 0; kh < kernel_h; ++kh) {
                        for (int kw = 0; kw < kernel_w; ++kw) {
                            int in_h = oh * stride_h - pad_h + kh;
                            int in_w = ow * stride_w - pad_w + kw;
                            if (in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width) {
                                int input_index = (in_h * in_width * in_channels) + (in_w * in_channels) + ic;
                                int weight_index = (((kh * kernel_w + kw) * in_channels + ic) * out_channels) + oc;
                                sum += inputs[input_index] * weights[weight_index];
                            }
                        }
                    }
                }
                sum += biases[oc];
                activation_function(outputs[(oh * out_width * out_channels) + (ow * out_channels) + oc], sum, alpha);
            }
        }
    }
}

template<typename Scalar, int out_channels, int out_height, int out_width>
void conv2DTransposeForward(Scalar* outputs, const Scalar* inputs, const Scalar* weights, const Scalar* biases,
                            int in_channels, int in_height, int in_width,
                            int kernel_h, int kernel_w, int stride_h, int stride_w,
                            int pad_h, int pad_w,
                            activationFunction<Scalar> activation_function, Scalar alpha) noexcept {
    // Simplified implementation for transposed convolution (stub)
    for (int i = 0; i < out_height * out_width * out_channels; ++i) {
        outputs[i] = 0;
    }
    // ... (proper transposed convolution implementation would go here)
    for (int i = 0; i < out_height * out_width * out_channels; ++i) {
        activation_function(outputs[i], outputs[i], alpha);
    }
}

template<typename Scalar, int out_size>
void conv1DForward(Scalar* outputs, const Scalar* inputs, const Scalar* weights, const Scalar* biases,
                   int in_size, int kernel_size, int stride, int pad,
                   activationFunction<Scalar> activation_function, Scalar alpha) noexcept {
    for (int o = 0; o < out_size; ++o) {
        Scalar sum = 0;
        for (int k = 0; k < kernel_size; ++k) {
            int in_index = o * stride - pad + k;
            if(in_index >= 0 && in_index < in_size){
                int weight_index = k * out_size + o;
                sum += inputs[in_index] * weights[weight_index];
            }
        }
        sum += biases[o];
        activation_function(outputs[o], sum, alpha);
    }
}

template<typename Scalar, int out_channels, int out_depth, int out_height, int out_width>
void conv3DForward(Scalar* outputs, const Scalar* inputs, const Scalar* weights, const Scalar* biases,
                   int in_channels, int in_depth, int in_height, int in_width,
                   int kernel_d, int kernel_h, int kernel_w, int stride_d, int stride_h, int stride_w,
                   int pad_d, int pad_h, int pad_w,
                   activationFunction<Scalar> activation_function, Scalar alpha) noexcept {
    // Simplified 3D convolution implementation
    for (int oc = 0; oc < out_channels; ++oc) {
        for (int od = 0; od < out_depth; ++od) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    Scalar sum = 0;
                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int kd = 0; kd < kernel_d; ++kd) {
                            for (int kh = 0; kh < kernel_h; ++kh) {
                                for (int kw = 0; kw < kernel_w; ++kw) {
                                    int in_d = od * stride_d - pad_d + kd;
                                    int in_h = oh * stride_h - pad_h + kh;
                                    int in_w = ow * stride_w - pad_w + kw;
                                    if(in_d >= 0 && in_d < in_depth && in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width){
                                        int input_index = ((in_d * in_height * in_width * in_channels) + (in_h * in_width * in_channels) + (in_w * in_channels) + ic);
                                        int weight_index = (((((kd * kernel_h + kh) * kernel_w + kw) * in_channels + ic) * out_channels) + oc);
                                        sum += inputs[input_index] * weights[weight_index];
                                    }
                                }
                            }
                        }
                    }
                    sum += biases[oc];
                    int output_index = ((od * out_height * out_width * out_channels) + (oh * out_width * out_channels) + (ow * out_channels) + oc);
                    activation_function(outputs[output_index], sum, alpha);
                }
            }
        }
    }
}

// template<typename Scalar, int out_channels, int out_height, int out_width>
// void depthwiseConv2DForward(Scalar* outputs, const Scalar* inputs, const Scalar* weights, const Scalar* biases,
//                             int in_channels, int in_height, int in_width,
//                             int kernel_h, int kernel_w, int stride_h, int stride_w,
//                             int pad_h, int pad_w,
//                             activationFunction<Scalar> activation_function, Scalar alpha) noexcept {
//     // Simplified depthwise convolution implementation (each input channel is convolved independently)
//     for (int c = 0; c < in_channels; ++c) {
//         for (int oh = 0; oh < out_height; ++oh) {
//             for (int ow = 0; ow < out_width; ++ow) {
//                 Scalar sum = 0;
//                 for (int kh = 0; kh < kernel_h; ++kh) {
//                     for (int kw = 0; kw < kernel_w; ++kw) {
//                         int in_h = oh * stride_h - pad_h + kh;
//                         int in_w = ow * stride_w - pad_w + kw;
//                         if(in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width){
//                             int input_index = (in_h * in_width * in_channels) + (in_w * in_channels) + c;
//                             int weight_index = (kh * kernel_w + kw) * in_channels + c;
//                             sum += inputs[input_index] * weights[weight_index];
//                         }
//                     }
//                 }
//                 sum += biases[c];
//                 int output_index = (oh * out_width * in_channels) + (ow * in_channels) + c;
//                 activation_function(outputs[output_index], sum, alpha);
//             }
//         }
//     }
// }

template<typename Scalar>
void depthwiseConv2DForward(Scalar* outputs, const Scalar* inputs, const Scalar* weights, const Scalar* biases,
                            int out_channels, int out_height, int out_width,
                            int in_channels, int in_height, int in_width,
                            int kernel_h, int kernel_w, int stride_h, int stride_w,
                            int pad_h, int pad_w,
                            activationFunction<Scalar> activation_function, Scalar alpha) noexcept {
    for (int c = 0; c < in_channels; ++c) {
        for (int oh = 0; oh < out_height; ++oh) {
            for (int ow = 0; ow < out_width; ++ow) {
                Scalar sum = 0;
                for (int kh = 0; kh < kernel_h; ++kh) {
                    for (int kw = 0; kw < kernel_w; ++kw) {
                        int in_h = oh * stride_h - pad_h + kh;
                        int in_w = ow * stride_w - pad_w + kw;
                        if (in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width) {
                            int input_index = (in_h * in_width * in_channels) + (in_w * in_channels) + c;
                            int weight_index = (kh * kernel_w + kw) * in_channels + c;
                            sum += inputs[input_index] * weights[weight_index];
                        }
                    }
                }
                sum += biases[c];
                int output_index = (oh * out_width * in_channels) + (ow * in_channels) + c;
                activation_function(outputs[output_index], sum, alpha);
            }
        }
    }
}


// template<typename Scalar, int out_channels, int out_height, int out_width>
// void separableConv2DForward(Scalar* outputs, const Scalar* inputs, const Scalar* depthwise_weights, const Scalar* pointwise_weights, const Scalar* biases,
//                             int in_channels, int in_height, int in_width,
//                             int kernel_h, int kernel_w, int stride_h, int stride_w,
//                             int pad_h, int pad_w,
//                             activationFunction<Scalar> activation_function, Scalar alpha) noexcept {
//     // First perform depthwise convolution (this is a simplified approach)
//     const int depthwise_output_size = in_height * in_width * in_channels; // assuming same spatial dims for simplicity
//     Scalar depthwise_output[depthwise_output_size];
//     depthwiseConv2DForward<Scalar, in_channels, in_height, in_width>(depthwise_output, inputs, depthwise_weights, biases,
//                                                                      in_channels, in_height, in_width,
//                                                                      kernel_h, kernel_w, stride_h, stride_w,
//                                                                      pad_h, pad_w, activation_function, 0.0);
//     // Then perform pointwise convolution
//     for (int oc = 0; oc < out_channels; ++oc) {
//         for (int i = 0; i < in_height * in_width; ++i) {
//             Scalar sum = 0;
//             for (int ic = 0; ic < in_channels; ++ic) {
//                 int index = i * in_channels + ic;
//                 int weight_index = ic * out_channels + oc;
//                 sum += depthwise_output[index] * pointwise_weights[weight_index];
//             }
//             sum += biases[oc];
//             outputs[i * out_channels + oc] = sum;
//             activation_function(outputs[i * out_channels + oc], sum, alpha);
//         }
//     }
// }

template<typename Scalar, int out_channels, int out_height, int out_width>
void separableConv2DForward(Scalar* outputs, const Scalar* inputs, const Scalar* depthwise_weights, const Scalar* pointwise_weights, const Scalar* biases,
                            int in_channels, int in_height, int in_width,
                            int kernel_h, int kernel_w, int stride_h, int stride_w,
                            int pad_h, int pad_w,
                            activationFunction<Scalar> activation_function, Scalar alpha) noexcept {
    // Use std::vector instead of VLA
    std::vector<Scalar> depthwise_output(in_height * in_width * in_channels, 0);

    // Call depthwiseConv2DForward with runtime parameters
    depthwiseConv2DForward(
        depthwise_output.data(), inputs, depthwise_weights, biases,
        in_channels, in_height, in_width,  // Pass out_channels as in_channels
        in_channels, in_height, in_width,  // Pass correct in_channels and dimensions
        kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
        activation_function, alpha);    

    // Then perform pointwise convolution
    for (int oc = 0; oc < out_channels; ++oc) {
        for (int i = 0; i < in_height * in_width; ++i) {
            Scalar sum = 0;
            for (int ic = 0; ic < in_channels; ++ic) {
                int index = i * in_channels + ic;
                int weight_index = ic * out_channels + oc;
                sum += depthwise_output[index] * pointwise_weights[weight_index];
            }
            sum += biases[oc];
            outputs[i * out_channels + oc] = sum;
            activation_function(outputs[i * out_channels + oc], sum, alpha);
        }
    }
}


template<typename Scalar>
void convLSTM2DForward(/* parameters */) noexcept {
    // Stub for ConvLSTM2D.
    // A full implementation would require handling time steps and cell states.
}

template<typename Scalar, int pool_height, int pool_width, int stride_h, int stride_w>
void maxPooling2D(Scalar* outputs, const Scalar* inputs, int in_height, int in_width, int channels) noexcept {
    // Calculate output dimensions (assumes no padding)
    int out_height = (in_height - pool_height) / stride_h + 1;
    int out_width = (in_width - pool_width) / stride_w + 1;
    for (int c = 0; c < channels; ++c) {
        for (int oh = 0; oh < out_height; ++oh) {
            for (int ow = 0; ow < out_width; ++ow) {
                Scalar max_val = -std::numeric_limits<Scalar>::infinity();
                for (int ph = 0; ph < pool_height; ++ph) {
                    for (int pw = 0; pw < pool_width; ++pw) {
                        int in_h = oh * stride_h + ph;
                        int in_w = ow * stride_w + pw;
                        int idx = (in_h * in_width * channels) + (in_w * channels) + c;
                        if (inputs[idx] > max_val) {
                            max_val = inputs[idx];
                        }
                    }
                }
                int out_idx = (oh * out_width * channels) + (ow * channels) + c;
                outputs[out_idx] = max_val;
            }
        }
    }
}

template<typename Scalar, int pool_height, int pool_width, int stride_h, int stride_w>
void avgPooling2D(Scalar* outputs, const Scalar* inputs, int in_height, int in_width, int channels) noexcept {
    int out_height = (in_height - pool_height) / stride_h + 1;
    int out_width = (in_width - pool_width) / stride_w + 1;
    for (int c = 0; c < channels; ++c) {
        for (int oh = 0; oh < out_height; ++oh) {
            for (int ow = 0; ow < out_width; ++ow) {
                Scalar sum = 0;
                for (int ph = 0; ph < pool_height; ++ph) {
                    for (int pw = 0; pw < pool_width; ++pw) {
                        int in_h = oh * stride_h + ph;
                        int in_w = ow * stride_w + pw;
                        int idx = (in_h * in_width * channels) + (in_w * channels) + c;
                        sum += inputs[idx];
                    }
                }
                int out_idx = (oh * out_width * channels) + (ow * channels) + c;
                outputs[out_idx] = sum / (pool_height * pool_width);
            }
        }
    }
}

template<typename Scalar>
void globalAvgPooling2D(Scalar* output, const Scalar* inputs, int in_height, int in_width, int channels) noexcept {
    // Compute global average per channel
    for (int c = 0; c < channels; ++c) {
        Scalar sum = 0;
        for (int h = 0; h < in_height; ++h) {
            for (int w = 0; w < in_width; ++w) {
                int idx = (h * in_width * channels) + (w * channels) + c;
                sum += inputs[idx];
            }
        }
        output[c] = sum / (in_height * in_width);
    }
}

template <typename Scalar = double>
auto cnn2(const std::array<std::array<std::array<Scalar, 1>, 8>, 8>& initial_input) {

        constexpr int flat_size = 64; // e.g. 64
        std::array<Scalar, flat_size> model_input;
        int idx = 0;
        for (int i0 = 0; i0 < 8; i0++) {
            for (int i1 = 0; i1 < 8; i1++) { // Fix the typo here
                for (int i2 = 0; i2 < 1; i2++) {
                    int flatIndex = i0 * 8 + i1 * 1 + i2 * 1;
                    model_input[flatIndex] = initial_input[i0][i1][i2];
                }
            }
        }

        // // check size
        // if (model_input.size() != flat_size) {
        //     throw std::invalid_argument("Invalid input size. Expected size: 64");
        // }
        // std::array<Scalar, 64> model_input = initial_input;

    if (model_input.size() != 64) { throw std::invalid_argument("Invalid input size. Expected size: 64"); }

    // --------------------------------------------------------------------------
    // Print out the old weights/biases for dense layers if needed
    // plus define the new conv (and pooling) dictionaries.
    // --------------------------------------------------------------------------

    // Layer 1 of type DepthwiseConv2D
    constexpr std::array<Scalar, 9> depthwiseKernel_1 = {4.791057110e-01, 1.078045089e-02, -2.177751660e-01, 5.264438391e-01, -1.350500733e-01, -4.456529766e-02, -3.912734687e-01, -5.210456848e-01, -7.342309691e-03};
    constexpr std::array<Scalar, 1> depthwiseBias_1 = {-2.083740037e-05};

    constexpr std::array<Scalar, 1> gamma_2 = {9.933543205e-01};
    constexpr std::array<Scalar, 1> beta_2 = {-1.060596481e-02};
    constexpr std::array<Scalar, 1> mean_2 = {-1.630094089e-02};
    constexpr std::array<Scalar, 1> variance_2 = {8.747540116e-01};
    constexpr Scalar epsilon_2 = 1.000000000e-03;

    // Layer 4 of type Conv2D
    constexpr std::array<Scalar, 8> convKernel_4 = {-3.384488523e-01, 8.180598728e-03, -1.237603743e-02, -6.497446299e-01, 2.065965533e-01, 2.091013044e-01, 4.104950726e-01, -6.879119277e-01};
    constexpr std::array<Scalar, 8> convBias_4 = {-6.868880155e-05, 1.835801377e-04, 9.248071583e-04, -1.925037213e-04, 4.759973963e-04, -2.119484125e-04, -7.063761004e-04, -7.829110837e-05};

    constexpr std::array<Scalar, 8> gamma_5 = {9.977204204e-01, 1.003378391e+00, 9.875956178e-01, 1.001788735e+00, 1.006754994e+00, 9.956783056e-01, 9.972378612e-01, 1.001842976e+00};
    constexpr std::array<Scalar, 8> beta_5 = {-3.261103760e-03, 5.636610091e-03, -1.256635785e-02, 3.614827758e-03, -1.137817185e-02, -8.364913054e-03, -2.582596848e-03, -5.586442538e-03};
    constexpr std::array<Scalar, 8> mean_5 = {-1.891811378e-02, 4.054055025e-04, -1.005842932e-03, -3.627588972e-02, 1.136499736e-02, 1.175613515e-02, 2.291415259e-02, -3.838510439e-02};
    constexpr std::array<Scalar, 8> variance_5 = {8.661125302e-01, 8.600608706e-01, 8.600775599e-01, 8.823249936e-01, 8.622338772e-01, 8.624158502e-01, 8.689885736e-01, 8.849931955e-01};
    constexpr Scalar epsilon_5 = 1.000000000e-03;

    // Layer 7 of type SeparableConv2D
    constexpr std::array<Scalar, 72> sepDepthwise_7 = {-1.865979284e-01, 1.477956772e-01, 5.306387693e-02, 9.534462541e-02, 2.228187174e-01, -1.110942196e-02, -2.609957159e-01, 2.270251662e-01, 1.146791577e-01, 2.087235451e-01, 1.465561837e-01, 1.448716521e-01, -1.679991744e-02, -9.926398844e-02, 1.618014127e-01, 2.288956195e-01, 1.353214234e-01, -2.052361518e-01, 1.284025759e-01, -8.544130251e-03, -1.739870310e-01, -2.060558647e-01, -4.336387664e-02, -2.106526792e-01, -1.209389418e-01, -7.795453072e-02, -1.825882494e-01, 2.688169181e-01, 7.153820246e-02, -1.960163713e-01, -1.874705404e-01, -1.692671329e-02, -1.523259282e-01, -1.412725896e-01, 2.060183734e-01, 2.521374822e-01, -4.508046806e-02, 1.319712996e-01, 2.321749926e-01, 1.807621121e-01, -9.242460132e-02, 6.900493056e-02, -2.471846342e-01, 2.307654470e-01, -1.705722660e-01, -1.170973405e-01, 2.391343117e-01, -2.508212924e-01, 2.383854091e-01, 9.168141335e-02, 6.925795972e-02, -2.278373539e-01, -2.071457058e-01, 1.083763614e-01, 1.893310100e-01, -1.141778380e-01, -2.852743119e-02, 4.604072124e-02, 1.237882003e-01, -1.909482330e-01, -4.989238828e-02, -8.400397003e-02, 8.920675516e-02, 1.345611662e-01, -6.956875324e-02, -1.168323830e-01, -9.321931005e-02, -1.308725476e-01, -9.860777110e-02, 9.714543819e-02, -2.235996723e-01, 5.086955428e-02};
    constexpr std::array<Scalar, 128> sepPointwise_7 = {-3.943703175e-01, 3.741484880e-01, -9.641176462e-02, 6.627850980e-02, -4.325694144e-01, -2.471377701e-01, 1.720011979e-01, 1.279311031e-01, -4.717490450e-02, -1.415318251e-01, -2.220559679e-02, -1.053925380e-01, -4.359023571e-01, 3.082570732e-01, 1.977714747e-01, -4.699331820e-01, 4.462367892e-01, -4.328263700e-01, -3.223749101e-01, 3.611817360e-01, -3.919341564e-01, -4.507321715e-01, 4.784313738e-01, -2.650538981e-01, 4.860058129e-01, 3.351662755e-01, -4.008086920e-01, 4.001659751e-01, 2.335315645e-01, -4.379780889e-01, -2.863761485e-01, 1.101177856e-01, 3.644542694e-01, 2.502288222e-01, 1.230695024e-01, 5.570581928e-02, -2.024086863e-01, -3.510543704e-01, -1.639596969e-01, 4.082275629e-01, 3.612112403e-01, 2.455277294e-01, 8.119721711e-02, 9.776343405e-02, 4.236234128e-01, -3.369370699e-01, -2.276303433e-02, -1.675456613e-01, 3.108967245e-01, 4.339919388e-01, -2.976138294e-01, 1.545170844e-01, -2.517434023e-02, -3.373855650e-01, 3.651584983e-01, 1.952223033e-01, -2.161895335e-01, 2.356867343e-01, -1.004194543e-01, -1.129771844e-01, -5.664777383e-02, -4.535152614e-01, 4.310549200e-01, 4.378164113e-01, 3.122232258e-01, 6.453078985e-02, 1.039594412e-01, -1.667640805e-01, 3.952890933e-01, 3.769375682e-01, -1.624051929e-01, 3.713676929e-01, -4.058728814e-01, -3.615449071e-01, -4.016778171e-01, -3.422538042e-01, -3.021961749e-01, 1.653328389e-01, -4.067851603e-01, -3.709194362e-01, 2.662095614e-02, 3.407799900e-01, 2.438402623e-01, 4.105296433e-01, -3.946924806e-01, 1.376646161e-01, 4.473998845e-01, 1.301633269e-01, -8.456061035e-02, 9.516350925e-02, 1.076957136e-01, 2.702864408e-01, -3.778245449e-01, 3.742979169e-01, -2.505472302e-01, 3.266809583e-01, 9.241836518e-02, -3.589209020e-01, -9.326241165e-02, 1.263671368e-01, -1.105744988e-01, 2.248224318e-01, -4.777083695e-01, 4.439808726e-01, 3.369101286e-01, 2.137498260e-01, 3.223895729e-01, 3.407678008e-01, 8.111818135e-02, -9.865590930e-02, -2.003399730e-01, 4.865674078e-01, 2.638141513e-01, -1.048326492e-01, -4.146257043e-01, 2.868884206e-01, -1.966398358e-01, 2.348201573e-01, -3.853877485e-01, 1.464249343e-01, 4.271564782e-01, 3.351082206e-01, 2.287002951e-01, -2.139195502e-01, 6.281935424e-02, 4.265775084e-01, -3.943045735e-01, -1.343842447e-01};
    constexpr std::array<Scalar, 16> sepPointwiseBias_7 = {6.692867828e-06, 2.347402187e-04, 9.128913371e-05, 2.058482551e-06, 1.133772312e-04, -1.284991304e-04, 2.352445790e-06, 1.605582947e-04, 2.047506314e-05, -4.730087312e-05, -4.674218508e-05, 5.035914364e-04, 2.127984917e-04, -9.539409075e-05, 9.575936565e-05, 1.483298256e-04};

    constexpr std::array<Scalar, 16> gamma_8 = {9.995368123e-01, 9.990187287e-01, 9.950011373e-01, 9.944267869e-01, 9.989464879e-01, 9.980170131e-01, 1.001037717e+00, 9.916695356e-01, 1.007386088e+00, 1.001566410e+00, 9.964578748e-01, 1.003946424e+00, 9.937018752e-01, 1.004101038e+00, 9.985683560e-01, 9.951828718e-01};
    constexpr std::array<Scalar, 16> beta_8 = {-4.312225617e-03, 5.308859050e-03, 3.266467247e-03, -6.956706755e-03, -7.900752244e-04, -9.034545161e-03, -4.551545251e-03, 3.614332527e-03, 8.326089010e-03, 9.774068370e-03, -1.104702987e-02, -4.147845320e-03, -1.053798478e-02, 3.656426910e-03, -6.424990017e-03, -8.039580658e-03};
    constexpr std::array<Scalar, 16> mean_8 = {1.055789553e-02, -2.713145223e-03, -1.976119913e-02, 9.343987331e-03, -7.575007621e-03, -1.265626680e-02, -7.342638448e-03, 5.059114192e-03, 1.869842596e-02, 2.503054403e-02, 1.544746570e-02, 5.444306880e-03, 1.837889478e-02, -2.081481740e-02, 1.448171306e-02, 2.686110325e-02};
    constexpr std::array<Scalar, 16> variance_8 = {8.612734079e-01, 8.654001951e-01, 8.618488312e-01, 8.625833988e-01, 8.634274006e-01, 8.619918823e-01, 8.674398661e-01, 8.626600504e-01, 8.671016693e-01, 8.649455905e-01, 8.660261631e-01, 8.651937246e-01, 8.629028797e-01, 8.625113964e-01, 8.626952767e-01, 8.721811771e-01};
    constexpr Scalar epsilon_8 = 1.000000000e-03;

    // Layer 10 of type SeparableConv2D
    constexpr std::array<Scalar, 144> sepDepthwise_10 = {-3.771617264e-02, 1.586123705e-01, 7.615466416e-02, -1.664630026e-01, -1.979031414e-01, -6.263080984e-02, 7.643008232e-02, 1.784743816e-01, -1.733304411e-01, 1.888602823e-01, -9.440016001e-02, 1.689088196e-01, -2.646436542e-02, -1.962775737e-02, 9.926639497e-02, 1.349247545e-01, 1.512927860e-01, 3.558492661e-02, 1.302355714e-02, 6.253875047e-02, 7.253338397e-02, 1.214979142e-01, 8.190640807e-02, -1.403503567e-01, -3.534004465e-02, 1.349188387e-01, 7.912289351e-02, 1.867171526e-01, 1.562011540e-01, -1.603973061e-01, 6.968799978e-02, 2.465715632e-02, -8.444955200e-02, -8.179242909e-02, 1.743333787e-01, -7.431871723e-03, 9.866931289e-02, 1.112959683e-01, 7.461334020e-02, 1.254745387e-02, -3.676633909e-02, -8.566746861e-02, 6.249718741e-02, 1.983677298e-01, 2.930486575e-02, 1.103938520e-01, 1.010264382e-01, -1.142458916e-01, 1.976609528e-01, 5.261142924e-02, -1.828182191e-01, -5.047571659e-02, -1.273089945e-01, 3.879543394e-02, -5.698865280e-02, -8.540388942e-02, -1.750904024e-01, -1.873743385e-01, -1.549244225e-01, 9.546700120e-02, 1.377967149e-01, -1.378502995e-01, -1.871083528e-01, -3.242213279e-02, -1.214686558e-01, 1.603059918e-01, -1.653948873e-01, 1.908080280e-01, 3.707847744e-02, 4.491947591e-02, -9.453205764e-02, -1.306568086e-01, 5.448525026e-02, 9.089212120e-02, 1.458788961e-01, 1.481660753e-01, 3.690299019e-02, 9.595131874e-02, 1.141952798e-01, -1.629500836e-01, -3.413200751e-02, 1.568738073e-01, -4.439224303e-02, -1.221584156e-01, 1.662188023e-01, -3.662264347e-02, -1.186917350e-01, -1.308513880e-01, 1.408606917e-01, 1.238673404e-01, -7.646175474e-02, 7.233786583e-02, -1.430528760e-01, 8.610753343e-03, 3.658381477e-02, -1.536023617e-01, -8.541145921e-02, 2.642202750e-02, 2.231898531e-02, 2.477178723e-02, 1.844080240e-01, -3.678021207e-02, -1.920427829e-01, -3.329223022e-02, -1.829283684e-01, 1.062051877e-01, 6.113474816e-02, -2.122344635e-02, -1.301110983e-01, -5.124251358e-03, -8.823347092e-02, 1.953181811e-02, -1.129275262e-01, 2.578485990e-03, -2.349578775e-02, 4.886929318e-02, -7.726567984e-02, 1.457696110e-01, 5.714377388e-03, -2.381366864e-02, -2.069751360e-02, -1.139095649e-01, -1.733260602e-01, -1.005167216e-01, 1.439905819e-02, 4.226735979e-02, -1.049052328e-01, -1.003248543e-01, -6.778648496e-02, 3.708431497e-02, -1.032627299e-01, -1.356097907e-01, 7.182599604e-02, -6.241112947e-02, -1.611914188e-01, -1.675482690e-01, 7.658018917e-02, -1.013388112e-01, 1.032577083e-01, 8.842700720e-02, 4.551157355e-02, -8.103634417e-02, 1.743678004e-01, -5.658793449e-02};
    constexpr std::array<Scalar, 256> sepPointwise_10 = {1.917102486e-01, 1.461259723e-01, 3.252052367e-01, 1.830216944e-01, 3.525715768e-01, -1.087072715e-01, 9.475661814e-02, 9.558028728e-02, 1.223318949e-01, -9.860952385e-03, -1.193196476e-01, 1.859820262e-02, -3.055092394e-01, 4.227309823e-01, 4.241010547e-02, -2.089650482e-01, -2.030164376e-02, 1.258833334e-02, 1.409981549e-01, 2.465333641e-01, 2.461139709e-01, 4.427607656e-01, -4.967503995e-02, -5.550302193e-02, 3.249305189e-01, -2.103945464e-01, 3.837375641e-01, -3.050777912e-01, 4.723880440e-02, -1.699351519e-01, -2.761662602e-01, 1.708081067e-01, 4.079443812e-01, -6.895670295e-02, -3.892226815e-01, -3.802302480e-01, 6.386347860e-02, -1.558811814e-01, 3.347644806e-01, -5.055366084e-02, 3.038400412e-01, -5.742587894e-02, -3.811986446e-01, 1.338051856e-01, 7.038029283e-02, -2.799175978e-01, 4.026407376e-02, -1.311429739e-01, 1.423081160e-01, -8.209085464e-02, 1.532416493e-01, 1.617079824e-01, -1.914959699e-01, -7.712939382e-02, -2.265907377e-01, 1.342898607e-01, -3.000466526e-01, -3.800790012e-01, -1.709441841e-01, 2.676426172e-01, 1.266840398e-01, 2.350437045e-01, 1.477706581e-01, -3.745023906e-01, 4.611323774e-02, -1.496699601e-01, -4.315171540e-01, 3.697928786e-01, -2.565581501e-01, -8.992674947e-02, 7.092195004e-02, 1.095005423e-01, 1.682039350e-01, 1.993460953e-01, 1.167497784e-01, 3.954223990e-01, -3.749258816e-01, 9.841491282e-02, 3.641532362e-01, -1.656642854e-01, 3.340311050e-01, -2.161832452e-01, -3.682437241e-01, 2.661343217e-01, -4.056447148e-01, 2.096792907e-01, -6.570012867e-02, -2.701435983e-02, 2.719554901e-01, -9.543304890e-02, 3.591350615e-01, 4.207745194e-01, 1.643672585e-01, 2.948130965e-01, 2.707759738e-01, 2.270807773e-01, 8.786929399e-02, 9.186901152e-02, 3.619429097e-02, -4.011410177e-01, 1.223737299e-01, 4.079861939e-01, -1.288259774e-01, -2.841378152e-01, 4.166879654e-01, -4.002585411e-01, -1.174717844e-01, 3.578110635e-01, 2.163997293e-01, 2.424939424e-01, 2.425002158e-01, 2.498409152e-02, -1.847729683e-01, -4.496526346e-02, 1.318595279e-02, -1.306714267e-01, -5.059283227e-02, 2.316714525e-01, 4.818103090e-02, 1.903369874e-01, -1.525358409e-01, -2.276127189e-01, 4.314411283e-01, -1.382393297e-03, 2.472312003e-01, -3.073438406e-01, 3.624068797e-01, -1.869666427e-01, -3.294453323e-01, -1.828679293e-01, 2.038476467e-01, 2.229955196e-01, 3.485930264e-01, -3.846590519e-01, 1.891909838e-01, 2.714738846e-01, -3.721468747e-01, 2.200042456e-01, -1.873740554e-01, 1.612514704e-01, -4.794102907e-02, -3.333898634e-02, 2.200712860e-01, -9.169831872e-02, 4.813547805e-02, 3.687951565e-01, -3.620871603e-01, -7.740012556e-02, 2.214795351e-01, 4.019297957e-01, -4.120542109e-01, 1.644877717e-02, -3.758453131e-01, -3.409609795e-01, 9.576383978e-02, -1.394811422e-01, -3.333300054e-01, 1.118368432e-01, -2.041934729e-01, 4.088388681e-01, -1.812191904e-01, 3.145728111e-01, 3.410665393e-01, -1.251356453e-01, 9.826084971e-02, 2.925030589e-01, 3.973486722e-01, 2.914494574e-01, 2.008563727e-01, -1.219039317e-02, 6.873467937e-03, 2.539317012e-01, 2.131249607e-01, 6.597195286e-03, -3.271221817e-01, 1.473467946e-01, -3.215661347e-01, -2.057358027e-01, 2.105838060e-01, 8.243438601e-02, -2.600490451e-01, 2.910005748e-01, 3.706366941e-02, 1.542256176e-01, 1.304252744e-01, -1.118303016e-01, 2.150658369e-01, 1.851912402e-02, -1.830876470e-01, 3.384218216e-01, 3.529486358e-01, -3.179850280e-01, -7.672363520e-02, -4.196007252e-01, 2.108845264e-01, 1.727568358e-01, 3.668719828e-01, 5.816524383e-03, 2.169013023e-01, -2.284605652e-01, -2.503901422e-01, 1.902857721e-01, 2.729899585e-01, 1.983906701e-02, -1.891821623e-01, -2.171493024e-01, -4.028745294e-01, 3.127447963e-01, -4.079003334e-01, -2.267833203e-01, 2.074870765e-01, -1.254218966e-01, -2.724687457e-01, -5.805538967e-03, -3.353739381e-01, 1.047090888e-01, -1.738778949e-01, -2.410021126e-01, 8.774662018e-02, 8.329242468e-02, 1.227731556e-01, -7.783490419e-02, 1.200664192e-01, 1.124807224e-01, 3.205099404e-01, 7.000648184e-04, -1.886017919e-01, -4.241927266e-01, -6.065089256e-02, -4.322192632e-03, 2.808156013e-01, -1.080533937e-01, 4.243006110e-01, 3.327183425e-01, -3.814651966e-01, -4.303476512e-01, -7.815538347e-02, -1.378305703e-01, 1.077692863e-02, -2.566843927e-01, -1.114132702e-01, 2.301536649e-01, 1.324748695e-01, -1.015447453e-01, 1.022894830e-01, 3.047534525e-01, -1.885780990e-01, -2.365468629e-02, -1.553794146e-01, 3.029009104e-01, -4.543586448e-02, 3.084249794e-01, -3.436426222e-01, -1.293003112e-01, 1.333420724e-01, 1.892257780e-01};
    constexpr std::array<Scalar, 16> sepPointwiseBias_10 = {-2.128137858e-04, 5.094251246e-04, -6.368365575e-05, -4.924530513e-04, -3.833316150e-04, 9.507687355e-05, 9.267561836e-06, -1.449749107e-04, 5.985127646e-04, 9.351818881e-05, 3.085552598e-04, 1.457223552e-04, 1.312021195e-04, 5.241130129e-04, 2.526186472e-05, 2.543295705e-05};

    constexpr std::array<Scalar, 16> gamma_11 = {9.893945456e-01, 9.936457276e-01, 9.940803051e-01, 9.898499846e-01, 9.969305992e-01, 1.012331128e+00, 9.913190603e-01, 1.006539583e+00, 9.865819216e-01, 9.871383309e-01, 1.003752947e+00, 1.003496289e+00, 9.867479205e-01, 9.967375994e-01, 9.860343933e-01, 9.937593341e-01};
    constexpr std::array<Scalar, 16> beta_11 = {-9.882653132e-03, -6.501554046e-03, -5.548204295e-03, -9.637972340e-03, -2.439140342e-03, 1.209408790e-02, -8.228344843e-03, 5.780874752e-03, -1.316933148e-02, -1.294082310e-02, 2.341107465e-03, 3.389762016e-03, -1.341732219e-02, -2.059475984e-03, -1.393211633e-02, -6.092781201e-03};
    constexpr std::array<Scalar, 16> mean_11 = {1.730349148e-03, -1.611777022e-02, -7.250561845e-03, 2.701438032e-02, -1.652650535e-02, 1.228309050e-02, 1.864222926e-03, -4.072531126e-03, 2.194871567e-02, 7.743681781e-04, 2.177114785e-02, -2.254217304e-02, -1.188866049e-02, 1.673878543e-02, -1.164209004e-02, 1.289699576e-03};
    constexpr std::array<Scalar, 16> variance_11 = {8.633658886e-01, 8.629000187e-01, 8.651255965e-01, 8.650976419e-01, 8.645109534e-01, 8.669048548e-01, 8.641960621e-01, 8.616101742e-01, 8.660686612e-01, 8.655049801e-01, 8.661211133e-01, 8.645362258e-01, 8.641509414e-01, 8.652913570e-01, 8.637973070e-01, 8.635177016e-01};
    constexpr Scalar epsilon_11 = 1.000000000e-03;

    // Layer 13 of type GlobalAveragePooling2D
    // Global average pooling layer for layer 13 (no extra parameters needed)

    // Dense or other layer 14
    constexpr std::array<Scalar, 80> weights_14 = {-2.593242824e-01, -1.817701617e-03, -2.243622094e-01, 5.540617183e-02, -3.547773659e-01, 2.232497931e-01, 4.942587912e-01, 2.675659060e-01, -2.854668796e-01, 4.675116539e-01, -1.242927834e-01, 6.819196511e-03, -1.655491889e-01, 2.965084910e-01, 9.190067649e-02, 2.348518223e-01, 4.120061994e-01, 2.381704301e-01, -2.898204625e-01, -3.152502477e-01, -3.612616360e-01, -1.551427841e-01, -3.450012207e-01, 4.535778463e-01, -1.745133996e-01, 3.493096232e-01, -4.591509402e-01, 1.801212430e-01, -2.207425237e-02, -6.352915615e-02, 3.330100179e-01, 3.624832332e-01, 2.281507254e-01, 2.383794188e-01, 1.793605536e-01, 6.489153951e-02, -7.115987688e-02, 4.436414540e-01, 8.742903173e-02, 4.160164893e-01, -2.457726002e-01, 4.924044311e-01, -1.796402633e-01, -1.882847846e-01, -4.156348705e-01, 8.175920695e-02, 4.515710473e-01, 4.605799615e-01, 4.179961681e-01, 1.939453036e-01, -3.019922674e-01, -2.795309126e-01, 3.146932125e-01, 1.130922660e-01, 3.288639784e-01, 2.362502962e-01, 2.809759229e-02, -4.288273156e-01, -4.865388870e-01, 1.458462561e-03, -3.914014995e-01, 4.421942234e-01, 9.422065318e-02, -1.021793783e-01, -5.545492843e-02, -1.794041395e-01, -7.232995331e-02, -4.631054103e-01, 4.324167967e-01, 6.820437312e-02, -2.545875311e-02, 3.860474825e-01, 2.318956703e-01, -2.549740952e-03, -1.874455959e-01, -4.551549554e-01, -9.686171263e-02, 2.129567266e-01, 3.922936916e-01, -2.017081082e-01};
    constexpr std::array<Scalar, 5> biases_14 = {1.124452706e-02, -1.256507356e-02, -1.260973746e-03, 1.542300452e-03, 3.139118198e-03};


    // Insert activation function definitions:

    auto relu = [](Scalar& output, Scalar input, Scalar alpha) noexcept {
        output = input > 0 ? input : 0;
    };

    auto sigmoid = [](Scalar& output, Scalar input, Scalar alpha) noexcept {
        output = 1 / (1 + std::exp(-input));
    };

    auto tanhCustom = [](Scalar& output, Scalar input, Scalar alpha) noexcept {
        output = std::tanh(input);
    };

    // OLD CODE:
    // auto leakyRelu = [](Scalar& output, Scalar input, Scalar alpha) noexcept {
    //     output = input > 0 ? input : alpha * input;
    // };
    // NEW CODE:
    auto leakyRelu = [](Scalar& output, Scalar input, Scalar alpha) noexcept {
        output = input > 0 ? input : alpha * input;
    };

    auto linear = [](Scalar& output, Scalar input, Scalar alpha) noexcept {
        output = input;
    };

    auto elu = [](Scalar& output, Scalar input, Scalar alpha) noexcept {
        output = input > 0 ? input : alpha * (std::exp(input) - 1);
    };

    // template<typename T> constexpr T SELU_LAMBDA = static_cast<T>(1.0507009873554804934193349852946);
    // template<typename T> constexpr T SELU_ALPHA = static_cast<T>(1.6732632423543772848170429916717);
    // auto selu = [](Scalar& output, Scalar input, Scalar alpha = SELU_ALPHA<double>) noexcept {
    //     using Scalar = decltype(input);
    //     output = SELU_LAMBDA<Scalar> * (input > 0 ? input : alpha * (std::exp(input) - 1));
    // };

    auto swish = [](Scalar& output, Scalar input, Scalar alpha) noexcept {
        output = input / (1 + std::exp(-alpha * input));
    };

    auto prelu = [](Scalar& output, Scalar input, Scalar alpha) noexcept {
        output = input > 0 ? input : alpha * input;
    };

    auto silu = [](Scalar& output, Scalar input, Scalar alpha) noexcept {
        auto sigmoid = 1 / (1 + std::exp(-input));
        output = input * sigmoid;
    };

    // --- NEW: Softmax lambda ---
    // This lambda computes the softmax of an input array of given size.
    // It takes pointers to the input and output arrays and the number of elements.
    auto softmax = [](const Scalar* input, Scalar* output, int size) noexcept {
        Scalar max_val = input[0];
        for (int i = 1; i < size; ++i) {
            if (input[i] > max_val)
                max_val = input[i];
        }
        Scalar sum = 0;
        for (int i = 0; i < size; ++i) {
            output[i] = std::exp(input[i] - max_val);
            sum += output[i];
        }
        for (int i = 0; i < size; ++i) {
            output[i] /= sum;
        }
    };

    // --------------------------------------------------------------------------
    // Now we do the actual forward pass logic.
    // For each layer i, based on the parameters, we call the appropriate function.

    // DepthwiseConv2D call for layer 1
    // std::array<Scalar, 64> layer_1_output;
    // depthwiseConv2DForward<Scalar, 1, 8, 8>(
    //     layer_1_output.data(), model_input.data(),
    //     depthwiseKernel_1.data(), depthwiseBias_1.data(),
    //     1, 8, 8, 3, 3, 1, 1, 1, 1, leakyRelu, 0.01);
    std::array<Scalar, 64> layer_1_output;
    depthwiseConv2DForward(
        layer_1_output.data(), model_input.data(),
        depthwiseKernel_1.data(), depthwiseBias_1.data(),
        1, 8, 8, 1, 8, 8, 3, 3, 1, 1, 1, 1,
        static_cast<activationFunction<Scalar>>(leakyRelu), 0.01);
    std::cout << "Layer 1 output: ";
    for (const auto& val : layer_1_output) std::cout << val << " ";
    std::cout << std::endl;


    std::array<Scalar, 1> layer_2_output;
    batchNormalization<Scalar, 1>(
        layer_2_output.data(), layer_1_output.data(),
        gamma_2.data(), beta_2.data(),
        mean_2.data(), variance_2.data(),
        epsilon_2);
    std::cout << "Layer 2 output: ";
    for (const auto& val : layer_2_output) std::cout << val << " ";
    std::cout << std::endl;

    // Pure activation layer 3
    std::array<Scalar, 1> layer_3_output;
    for (int i = 0; i < 1; ++i) {
        linear(layer_3_output[i], layer_2_output[i], 0.0);
    }
    std::cout << "Layer 3 output: ";
    for (const auto& val : layer_3_output) std::cout << val << " ";
    std::cout << std::endl;

    // // conv2DForward call for layer 4
    // std::array<Scalar, 8> layer_4_output;
    // conv2DForward<Scalar, 8, 8, 8>(
    //     layer_4_output.data(), layer_3_output.data(),
    //     convKernel_4.data(), convBias_4.data(),
    //     1, 8, 8, 3, 3, 1, 1, 1, 1, leakyRelu, 0.01);
    std::array<Scalar, 8 * 8 * 8> layer_4_output; // Ensure correct size
    conv2DForward<Scalar, 8, 8, 8>(
        layer_4_output.data(), layer_3_output.data(),
        convKernel_4.data(), convBias_4.data(),
        1, 8, 8,       // in_channels, in_height, in_width
        3, 3,          // kernel_h, kernel_w
        1, 1,          // stride_h, stride_w
        1, 1,          // pad_h, pad_w
        static_cast<activationFunction<Scalar>>(leakyRelu), 0.01);
    std::cout << "Layer 4 output: ";
    for (const auto& val : layer_4_output) std::cout << val << " ";
    std::cout << std::endl;
    
    


    std::array<Scalar, 8> layer_5_output;
    batchNormalization<Scalar, 8>(
        layer_5_output.data(), layer_4_output.data(),
        gamma_5.data(), beta_5.data(),
        mean_5.data(), variance_5.data(),
        epsilon_5);
    std::cout << "Layer 5 output: ";
    for (const auto& val : layer_5_output) std::cout << val << " ";
    std::cout << std::endl;

    // Pure activation layer 6
    std::array<Scalar, 8> layer_6_output;
    for (int i = 0; i < 8; ++i) {
        linear(layer_6_output[i], layer_5_output[i], 0.0);
    }
    std::cout << "Layer 6 output: ";
    for (const auto& val : layer_6_output) std::cout << val << " ";
    std::cout << std::endl;

    // // SeparableConv2D call for layer 7
    // std::array<Scalar, 16> layer_7_output;
    // separableConv2DForward<Scalar, 16, 8, 8>(
    //     layer_7_output.data(), layer_6_output.data(),
    //     sepDepthwise_7.data(), sepPointwise_7.data(), sepPointwiseBias_7.data(),
    //     8, 8, 8, 3, 3, 1, 1, 1, 1, leakyRelu, 0.01);
    std::array<Scalar, 16 * 8 * 8> layer_7_output; // Ensure correct size
    separableConv2DForward<Scalar, 16, 8, 8>(
        layer_7_output.data(), layer_6_output.data(),
        sepDepthwise_7.data(), sepPointwise_7.data(), sepPointwiseBias_7.data(),
        8, 8, 8,       // in_channels, in_height, in_width
        3, 3,          // kernel_h, kernel_w
        1, 1,          // stride_h, stride_w
        1, 1,          // pad_h, pad_w
        static_cast<activationFunction<Scalar>>(leakyRelu), 0.01);
    std::cout << "Layer 7 output: ";
    for (const auto& val : layer_7_output) std::cout << val << " ";
    std::cout << std::endl;
    
    


    std::array<Scalar, 16> layer_8_output;
    batchNormalization<Scalar, 16>(
        layer_8_output.data(), layer_7_output.data(),
        gamma_8.data(), beta_8.data(),
        mean_8.data(), variance_8.data(),
        epsilon_8);
    std::cout << "Layer 8 output: ";
    for (const auto& val : layer_8_output) std::cout << val << " ";
    std::cout << std::endl;

    // Pure activation layer 9
    std::array<Scalar, 16> layer_9_output;
    for (int i = 0; i < 16; ++i) {
        linear(layer_9_output[i], layer_8_output[i], 0.0);
    }
    std::cout << "Layer 9 output: ";
    for (const auto& val : layer_9_output) std::cout << val << " ";
    std::cout << std::endl;

    // // SeparableConv2D call for layer 10
    // std::array<Scalar, 16> layer_10_output;
    // separableConv2DForward<Scalar, 16, 8, 8>(
    //     layer_10_output.data(), layer_9_output.data(),
    //     sepDepthwise_10.data(), sepPointwise_10.data(), sepPointwiseBias_10.data(),
    //     16, 8, 8, 3, 3, 1, 1, 1, 1, leakyRelu, 0.01);
    std::array<Scalar, 16 * 8 * 8> layer_10_output; // Ensure correct size
    separableConv2DForward<Scalar, 16, 8, 8>(
        layer_10_output.data(), layer_9_output.data(),
        sepDepthwise_10.data(), sepPointwise_10.data(), sepPointwiseBias_10.data(),
        16, 8, 8,      // in_channels, in_height, in_width
        3, 3,          // kernel_h, kernel_w
        1, 1,          // stride_h, stride_w
        1, 1,          // pad_h, pad_w
        static_cast<activationFunction<Scalar>>(leakyRelu), 0.01);
    std::cout << "Layer 10 output: ";
    for (const auto& val : layer_10_output) std::cout << val << " ";
    std::cout << std::endl;
    
    

    std::array<Scalar, 16> layer_11_output;
    batchNormalization<Scalar, 16>(
        layer_11_output.data(), layer_10_output.data(),
        gamma_11.data(), beta_11.data(),
        mean_11.data(), variance_11.data(),
        epsilon_11);
    std::cout << "Layer 11 output: ";
    for (const auto& val : layer_11_output) std::cout << val << " ";
    std::cout << std::endl;

    // Pure activation layer 12
    std::array<Scalar, 16> layer_12_output;
    for (int i = 0; i < 16; ++i) {
        linear(layer_12_output[i], layer_11_output[i], 0.0);
    }
    std::cout << "Layer 12 output: ";
    for (const auto& val : layer_12_output) std::cout << val << " ";
    std::cout << std::endl;

    // GlobalAveragePooling2D call for layer 13
    std::array<Scalar, 16> layer_13_output;
    globalAvgPooling2D<Scalar>(
        layer_13_output.data(), layer_12_output.data(),
        8, 8, 16);
    std::cout << "Layer 13 output: ";
    for (const auto& val : layer_13_output) std::cout << val << " ";
    std::cout << std::endl;

    // std::array<Scalar, 5> layer_14_output;
    // forwardPass<Scalar, 5>(
    //     layer_14_output.data(), layer_13_output.data(),
    //     weights_14.data(), biases_14.data(),
    //     16, linear, 0.0);
        // --- Dense layer (pre-softmax) for layer 14 ---
    std::array<Scalar, 5> dense_output;
    forwardPass<Scalar, 5>(
        dense_output.data(), layer_13_output.data(),
        weights_14.data(), biases_14.data(),
        16, linear, 0.0);

    // --- Apply softmax to dense_output ---
    std::array<Scalar, 5> layer_14_output;
    softmax(dense_output.data(), layer_14_output.data(), 5);


    std::cout << "Layer 14 output: ";
    for (const auto& val : layer_14_output) std::cout << val << " ";
    std::cout << std::endl;

    // final placeholder
    std::array<Scalar, 10> model_output{}; // example
    return model_output;
}
