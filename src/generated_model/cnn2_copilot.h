#pragma once
#include <iostream>
#include <array>
#include <random>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>

// Activation function type
template<typename Scalar>
using activationFunction = void(*)(Scalar&, Scalar, Scalar);

//
// Manipulation functions: normalization and forward pass
//
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
    for (int i = 0; i < output_size; ++i) {
        Scalar sum = 0;
        for (int j = 0; j < input_size; ++j) {
            sum += inputs[j] * weights[j * output_size + i];
        }
        sum += biases[i];
        activation_function(outputs[i], sum, alpha);
    }
}

//
// Convolution functions
//

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
                int output_index = (oh * out_width * out_channels) + (ow * out_channels) + oc;
                activation_function(outputs[output_index], sum, alpha);
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
    int total = out_height * out_width * out_channels;
    for (int i = 0; i < total; ++i) {
        outputs[i] = 0;
    }
    for (int i = 0; i < total; ++i) {
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
            if (in_index >= 0 && in_index < in_size) {
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
                                    if (in_d >= 0 && in_d < in_depth && in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width) {
                                        int input_index = ((in_d * in_height * in_width * in_channels) +
                                                           (in_h * in_width * in_channels) +
                                                           (in_w * in_channels) + ic);
                                        int weight_index = (((((kd * kernel_h + kh) * kernel_w + kw) * in_channels + ic) * out_channels) + oc);
                                        sum += inputs[input_index] * weights[weight_index];
                                    }
                                }
                            }
                        }
                    }
                    sum += biases[oc];
                    int output_index = ((od * out_height * out_width * out_channels) +
                                        (oh * out_width * out_channels) +
                                        (ow * out_channels) + oc);
                    activation_function(outputs[output_index], sum, alpha);
                }
            }
        }
    }
}

// -----------------------------------------------------------------
// Forward declaration of depthwiseConv2DForward so that separableConv2DForward can use it.
template<typename Scalar, int out_channels, int out_height, int out_width>
void depthwiseConv2DForward(Scalar* outputs, const Scalar* inputs, const Scalar* weights, const Scalar* biases,
                            int in_channels, int in_height, int in_width,
                            int kernel_h, int kernel_w, int stride_h, int stride_w,
                            int pad_h, int pad_w,
                            activationFunction<Scalar> activation_function, Scalar alpha) noexcept;

// Definition of depthwiseConv2DForward:
template<typename Scalar, int out_channels, int out_height, int out_width>
void depthwiseConv2DForward(Scalar* outputs, const Scalar* inputs, const Scalar* weights, const Scalar* biases,
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

// -----------------------------------------------------------------
// separableConv2DForward now takes non-type template parameters for in_channels, in_height, in_width.
template<typename Scalar, int out_channels, int out_height, int out_width, int in_channels, int in_height, int in_width>
void separableConv2DForward(Scalar* outputs, const Scalar* inputs, const Scalar* depthwise_weights, const Scalar* pointwise_weights, const Scalar* biases,
                            int kernel_h, int kernel_w, int stride_h, int stride_w,
                            int pad_h, int pad_w,
                            activationFunction<Scalar> activation_function, Scalar alpha) noexcept {
    constexpr int depthwise_output_size = in_height * in_width * in_channels;
    std::array<Scalar, depthwise_output_size> depthwise_output{};
    depthwiseConv2DForward<Scalar, in_channels, in_height, in_width>(
        depthwise_output.data(), inputs, depthwise_weights, biases,
        in_channels, in_height, in_width, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
        [](Scalar& o, Scalar i, Scalar a) noexcept { o = i; }, 0.0);
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
}

//
// Inline lambda activation functions
//
inline auto linear_lambda = [](auto& output, auto input, auto alpha) noexcept {
    output = input;
};

inline auto relu_lambda = [](auto& output, auto input, auto alpha) noexcept {
    output = input > 0 ? input : 0;
};

inline auto sigmoid_lambda = [](auto& output, auto input, auto alpha) noexcept {
    output = 1 / (1 + std::exp(-input));
};

inline auto tanhCustom_lambda = [](auto& output, auto input, auto alpha) noexcept {
    output = std::tanh(input);
};

//
// Generated CNN function (cnn2)
// This function accepts an input with raw shape [1][8][8][1],
// flattens it, processes it through several layers, and returns a 1D output.
//
template <typename Scalar = double>
auto cnn2(const std::array<std::array<std::array<std::array<Scalar, 1>, 8>, 8>, 1>& initial_input) {
    // Flatten the 4D input (assume batch size 1) into a flat array of size 8*8*1 = 64.
    constexpr int flat_size = 8 * 8 * 1;
    std::array<Scalar, flat_size> flat_input;
    for (int i0 = 0; i0 < 8; i0++) {
        for (int i1 = 0; i1 < 8; i1++) {
            for (int i2 = 0; i2 < 1; i2++) {
                flat_input[i0 * 8 * 1 + i1 * 1 + i2] = initial_input[0][i0][i1][i2];
            }
        }
    }
    auto model_input = flat_input;
    if (model_input.size() != flat_size) {
        throw std::invalid_argument("Invalid input size. Expected size: 64");
    }

    //
    // Long arrays for weights, biases, normalization parameters, etc.
    //
    constexpr std::array<Scalar, 9> weights_1 = {4.791057110e-01, 1.078045089e-02, -2.177751660e-01, 5.264438391e-01, -1.350500733e-01, -4.456529766e-02, -3.912734687e-01, -5.210456848e-01, -7.342309691e-03};
    constexpr std::array<Scalar, 1> biases_1 = {-2.083740037e-05};

    constexpr std::array<Scalar, 1> gamma_2 = {9.933543205e-01};
    constexpr std::array<Scalar, 1> beta_2 = {-1.060596481e-02};
    constexpr std::array<Scalar, 1> mean_2 = {-1.630094089e-02};
    constexpr std::array<Scalar, 1> variance_2 = {8.747540116e-01};
    constexpr Scalar epsilon_2 = 1.000000000e-03;

    constexpr std::array<Scalar, 8> weights_4 = {-3.384488523e-01, 8.180598728e-03, -1.237603743e-02, -6.497446299e-01, 2.065965533e-01, 2.091013044e-01, 4.104950726e-01, -6.879119277e-01};
    constexpr std::array<Scalar, 8> biases_4 = {-6.868880155e-05, 1.835801377e-04, 9.248071583e-04, -1.925037213e-04, 4.759973963e-04, -2.119484125e-04, -7.063761004e-04, -7.829110837e-05};

    constexpr std::array<Scalar, 8> gamma_5 = {9.977204204e-01, 1.003378391e+00, 9.875956178e-01, 1.001788735e+00, 1.006754994e+00, 9.956783056e-01, 9.972378612e-01, 1.001842976e+00};
    constexpr std::array<Scalar, 8> beta_5 = {-3.261103760e-03, 5.636610091e-03, -1.256635785e-02, 3.614827758e-03, -1.137817185e-02, -8.364913054e-03, -2.582596848e-03, -5.586442538e-03};
    constexpr std::array<Scalar, 8> mean_5 = {-1.891811378e-02, 4.054055025e-04, -1.005842932e-03, -3.627588972e-02, 1.136499736e-02, 1.175613515e-02, 2.291415259e-02, -3.838510439e-02};
    constexpr std::array<Scalar, 8> variance_5 = {8.661125302e-01, 8.600608706e-01, 8.600775599e-01, 8.823249936e-01, 8.622338772e-01, 8.624158502e-01, 8.689885736e-01, 8.849931955e-01};
    constexpr Scalar epsilon_5 = 1.000000000e-03;

    // Dummy placeholders for weights_7 and biases_7:
    constexpr std::array<Scalar, 9> weights_7 = {0};
    constexpr std::array<Scalar, 9> biases_7 = {0};

    constexpr std::array<Scalar, 16> gamma_8 = {9.995368123e-01, 9.990187287e-01, 9.950011373e-01, 9.944267869e-01, 9.989464879e-01, 9.980170131e-01, 1.001037717e+00, 9.916695356e-01, 1.007386088e+00, 1.001566410e+00, 9.964578748e-01, 1.003946424e+00, 9.937018752e-01, 1.004101038e+00, 9.985683560e-01, 9.951828718e-01};
    constexpr std::array<Scalar, 16> beta_8 = {-4.312225617e-03, 5.308859050e-03, 3.266467247e-03, -6.956706755e-03, -7.900752244e-04, -9.034545161e-03, -4.551545251e-03, 3.614332527e-03, 8.326089010e-03, 9.774068370e-03, -1.104702987e-02, -4.147845320e-03, -1.053798478e-02, 3.656426910e-03, -6.424990017e-03, -8.039580658e-03};
    constexpr std::array<Scalar, 16> mean_8 = {1.055789553e-02, -2.713145223e-03, -1.976119913e-02, 9.343987331e-03, -7.575007621e-03, -1.265626680e-02, -7.342638448e-03, 5.059114192e-03, 1.869842596e-02, 2.503054403e-02, 1.544746570e-02, 5.444306880e-03, 1.837889478e-02, -2.081481740e-02, 1.448171306e-02, 2.686110325e-02};
    constexpr std::array<Scalar, 16> variance_8 = {8.612734079e-01, 8.654001951e-01, 8.618488312e-01, 8.625833988e-01, 8.634274006e-01, 8.619918823e-01, 8.674398661e-01, 8.626600504e-01, 8.671016693e-01, 8.649455905e-01, 8.660261631e-01, 8.651937246e-01, 8.629028797e-01, 8.625113964e-01, 8.626952767e-01, 8.721811771e-01};
    constexpr Scalar epsilon_8 = 1.000000000e-03;

    // Dummy placeholders for weights_10 and biases_10:
    constexpr std::array<Scalar, 16> weights_10 = {0};
    constexpr std::array<Scalar, 16> biases_10 = {0};

    constexpr std::array<Scalar, 16> gamma_11 = {9.893945456e-01, 9.936457276e-01, 9.940803051e-01, 9.898499846e-01, 9.969305992e-01, 1.012331128e+00, 9.913190603e-01, 1.006539583e+00, 9.865819216e-01, 9.871383309e-01, 1.003752947e+00, 1.003496289e+00, 9.867479205e-01, 9.967375994e-01, 9.860343933e-01, 9.937593341e-01};
    constexpr std::array<Scalar, 16> beta_11 = {-9.882653132e-03, -6.501554046e-03, -5.548204295e-03, -9.637972340e-03, -2.439140342e-03, 1.209408790e-02, -8.228344843e-03, 5.780874752e-03, -1.316933148e-02, -1.294082310e-02, 2.341107465e-03, 3.389762016e-03, -1.341732219e-02, -2.059475984e-03, -1.393211633e-02, -6.092781201e-03};
    constexpr std::array<Scalar, 16> mean_11 = {1.730349148e-03, -1.611777022e-02, -7.250561845e-03, 2.701438032e-02, -1.652650535e-02, 1.228309050e-02, 1.864222926e-03, -4.072531126e-03, 2.194871567e-02, 7.743681781e-04, 2.177114785e-02, -2.254217304e-02, -1.188866049e-02, 1.673878543e-02, -1.164209004e-02, 1.289699576e-03};
    constexpr std::array<Scalar, 16> variance_11 = {8.633658886e-01, 8.629000187e-01, 8.651255965e-01, 8.650976419e-01, 8.645109534e-01, 8.669048548e-01, 8.641960621e-01, 8.616101742e-01, 8.660686612e-01, 8.655049801e-01, 8.661211133e-01, 8.645362258e-01, 8.641509414e-01, 8.652913570e-01, 8.637973070e-01, 8.635177016e-01};
    constexpr Scalar epsilon_11 = 1.000000000e-03;

    constexpr std::array<Scalar, 80> weights_14 = {-2.593242824e-01, -1.817701617e-03, -2.243622094e-01, 5.540617183e-02, -3.547773659e-01, 2.232497931e-01, 4.942587912e-01, 2.675659060e-01, -2.854668796e-01, 4.675116539e-01, -1.242927834e-01, 6.819196511e-03, -1.655491889e-01, 2.965084910e-01, 9.190067649e-02, 2.348518223e-01, 4.120061994e-01, 2.381704301e-01, -2.898204625e-01, -3.152502477e-01, -3.612616360e-01, -1.551427841e-01, -3.450012207e-01, 4.535778463e-01, -1.745133996e-01, 3.493096232e-01, -4.591509402e-01, 1.801212430e-01, -2.207425237e-02, -6.352915615e-02, 3.330100179e-01, 3.624832332e-01, 2.281507254e-01, 2.383794188e-01, 1.793605536e-01, 6.489153951e-02, -7.115987688e-02, 4.436414540e-01, 8.742903173e-02, 4.160164893e-01, -2.457726002e-01, 4.924044311e-01, -1.796402633e-01, -1.882847846e-01, -4.156348705e-01, 8.175920695e-02, 4.515710473e-01, 4.605799615e-01, 4.179961681e-01, 1.939453036e-01, -3.019922674e-01, -2.795309126e-01, 3.146932125e-01, 1.130922660e-01, 3.288639784e-01, 2.362502962e-01, 2.809759229e-02, -4.288273156e-01, -4.865388870e-01, 1.458462561e-03, -3.914014995e-01, 4.421942234e-01, 9.422065318e-02, -1.021793783e-01, -5.545492843e-02, -1.794041395e-01, -7.232995331e-02, -4.631054103e-01, 4.324167967e-01, 6.820437312e-02, -2.545875311e-02, 3.860474825e-01, 2.318956703e-01, -2.549740952e-03, -1.874455959e-01, -4.551549554e-01, -9.686171263e-02, 2.129567266e-01, 3.922936916e-01, -2.017081082e-01};
    constexpr std::array<Scalar, 5> biases_14 = {1.124452706e-02, -1.256507356e-02, -1.260973746e-03, 1.542300452e-03, 3.139118198e-03};

    //
    // Inline lambda activation functions
    //
    auto linear = linear_lambda;
    auto relu = relu_lambda;
    auto sigmoid = sigmoid_lambda;
    auto tanhCustom = tanhCustom_lambda;

    //
    // Processing layers
    //

    // Layer 1: Depthwise Convolution using weights_1 & biases_1
    // Assumed input dimensions: in_channels = 1, in_height = 8, in_width = 8, kernel 3x3, stride 1, pad 1.
    constexpr int in_channels = 1, in_height = 8, in_width = 8, kernel_h = 3, kernel_w = 3, stride_h = 1, stride_w = 1, pad_h = 1, pad_w = 1;
    // For simplicity, assume output size equals input size.
    std::array<Scalar, 8 * 8 * 1> depthwise_output;
    conv2DForward<Scalar, 1, 8, 8>(
        depthwise_output.data(), model_input.data(), weights_1.data(), biases_1.data(),
        in_channels, in_height, in_width, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
        linear, 0.0);

    std::array<Scalar, 1> layer_2_output;
    batchNormalization<Scalar, 1>(layer_2_output.data(), depthwise_output.data(), gamma_2.data(), beta_2.data(), mean_2.data(), variance_2.data(), epsilon_2);

    std::array<Scalar, 1> layer_3_output;
    linear(layer_3_output[0], layer_2_output[0], 0.0);

    // Layer 4: Standard 2D Convolution
    constexpr int in_height_4 = 8, in_width_4 = 8, in_channels_4 = 1, kernel_h_4 = 1, kernel_w_4 = 1, stride_h_4 = 1, stride_w_4 = 1, pad_h_4 = 0, pad_w_4 = 0;
    constexpr int out_height_4 = (in_height_4 + 2 * pad_h_4 - kernel_h_4) / stride_h_4 + 1;
    constexpr int out_width_4 = (in_width_4 + 2 * pad_w_4 - kernel_w_4) / stride_w_4 + 1;
    std::array<Scalar, out_height_4 * out_width_4 * 8> layer_4_output;
    conv2DForward<Scalar, 8, out_height_4, out_width_4>(
        layer_4_output.data(), layer_3_output.data(), weights_4.data(), biases_4.data(),
        in_channels_4, in_height_4, in_width_4, kernel_h_4, kernel_w_4, stride_h_4, stride_w_4, pad_h_4, pad_w_4,
        linear, 0.0);

    std::array<Scalar, 8> layer_5_output;
    batchNormalization<Scalar, 8>(layer_5_output.data(), layer_4_output.data(), gamma_5.data(), beta_5.data(), mean_5.data(), variance_5.data(), epsilon_5);

    std::array<Scalar, 8> layer_6_output;
    for (int i = 0; i < 8; i++) {
        linear(layer_6_output[i], layer_5_output[i], 0.0);
    }

    // Layer 7: Separable Convolution using dummy weights_7 & biases_7
    constexpr int in_channels_7 = 1, in_height_7 = 8, in_width_7 = 8, kernel_h_7 = 3, kernel_w_7 = 3, stride_h_7 = 1, stride_w_7 = 1, pad_h_7 = 1, pad_w_7 = 1;
    constexpr int out_height_7 = (in_height_7 + 2 * pad_h_7 - kernel_h_7) / stride_h_7 + 1;
    constexpr int out_width_7 = (in_width_7 + 2 * pad_w_7 - kernel_w_7) / stride_w_7 + 1;
    std::array<Scalar, out_height_7 * out_width_7 * 16> layer_7_output;
    separableConv2DForward<Scalar, 16, out_height_7, out_width_7, in_channels_7, in_height_7, in_width_7>(
        layer_7_output.data(), layer_6_output.data(), 
        weights_7.data(), weights_7.data(), biases_7.data(),
        kernel_h_7, kernel_w_7, stride_h_7, stride_w_7, pad_h_7, pad_w_7,
        linear, 0.0);

    std::array<Scalar, 16> layer_8_output;
    batchNormalization<Scalar, 16>(layer_8_output.data(), layer_7_output.data(), gamma_8.data(), beta_8.data(), mean_8.data(), variance_8.data(), epsilon_8);

    std::array<Scalar, 16> layer_9_output;
    for (int i = 0; i < 16; i++) {
        linear(layer_9_output[i], layer_8_output[i], 0.0);
    }

    // Layer 10: Separable Convolution using dummy weights_10 & biases_10
    constexpr int in_channels_10 = 1, in_height_10 = 8, in_width_10 = 8, kernel_h_10 = 3, kernel_w_10 = 3, stride_h_10 = 1, stride_w_10 = 1, pad_h_10 = 1, pad_w_10 = 1;
    constexpr int out_height_10 = (in_height_10 + 2 * pad_h_10 - kernel_h_10) / stride_h_10 + 1;
    constexpr int out_width_10 = (in_width_10 + 2 * pad_w_10 - kernel_w_10) / stride_w_10 + 1;
    std::array<Scalar, out_height_10 * out_width_10 * 16> layer_10_output;
    separableConv2DForward<Scalar, 16, out_height_10, out_width_10, in_channels_10, in_height_10, in_width_10>(
        layer_10_output.data(), layer_9_output.data(),
        weights_10.data(), weights_10.data(), biases_10.data(),
        kernel_h_10, kernel_w_10, stride_h_10, stride_w_10, pad_h_10, pad_w_10,
        linear, 0.0);

    std::array<Scalar, 16> layer_11_output;
    batchNormalization<Scalar, 16>(layer_11_output.data(), layer_10_output.data(), gamma_11.data(), beta_11.data(), mean_11.data(), variance_11.data(), epsilon_11);

    std::array<Scalar, 16> layer_12_output;
    for (int i = 0; i < 16; i++) {
        linear(layer_12_output[i], layer_11_output[i], 0.0);
    }

    std::array<Scalar, 16> layer_13_output;
    for (int i = 0; i < 16; i++) {
        linear(layer_13_output[i], layer_12_output[i], 0.0);
    }

    std::array<Scalar, 5> layer_14_output;
    forwardPass<Scalar, 5>(layer_14_output.data(), layer_13_output.data(), weights_14.data(), biases_14.data(), 16, linear, 0.0);

    auto model_output = layer_14_output;
    return model_output;
}

// Function to get intermediate values for debugging
template <typename Scalar>
std::vector<std::vector<Scalar>> get_intermediate_values() {
    std::vector<std::vector<Scalar>> intermediate_values;
    // Add code to collect intermediate values after each layer
    // Example:
    // intermediate_values.push_back(layer1_output);
    // intermediate_values.push_back(layer2_output);
    // ...
    return intermediate_values;
}

template <typename Scalar>
std::vector<std::vector<Scalar>> intermediate_values;

template <typename Scalar>
void log_intermediate_values(const std::vector<Scalar>& values) {
    intermediate_values<Scalar>.push_back(values);
}

template <typename Scalar = double>
auto cnn2(const std::array<std::array<std::array<std::array<Scalar, 1>, 8>, 8>, 1>& initial_input) {
    // Flatten the 4D input (assume batch size 1) into a flat array of size 8*8*1 = 64.
    constexpr int flat_size = 8 * 8 * 1;
    std::array<Scalar, flat_size> flat_input;
    for (int i0 = 0; i0 < 8; i0++) {
        for (int i1 = 0; i1 < 8; i1++) {
            for (int i2 = 0; i2 < 1; i2++) {
                flat_input[i0 * 8 * 1 + i1 * 1 + i2] = initial_input[0][i0][i1][i2];
            }
        }
    }
    auto model_input = flat_input;
    if (model_input.size() != flat_size) {
        throw std::invalid_argument("Invalid input size. Expected size: 64");
    }

    //
    // Long arrays for weights, biases, normalization parameters, etc.
    //
    constexpr std::array<Scalar, 9> weights_1 = {4.791057110e-01, 1.078045089e-02, -2.177751660e-01, 5.264438391e-01, -1.350500733e-01, -4.456529766e-02, -3.912734687e-01, -5.210456848e-01, -7.342309691e-03};
    constexpr std::array<Scalar, 1> biases_1 = {-2.083740037e-05};

    constexpr std::array<Scalar, 1> gamma_2 = {9.933543205e-01};
    constexpr std::array<Scalar, 1> beta_2 = {-1.060596481e-02};
    constexpr std::array<Scalar, 1> mean_2 = {-1.630094089e-02};
    constexpr std::array<Scalar, 1> variance_2 = {8.747540116e-01};
    constexpr Scalar epsilon_2 = 1.000000000e-03;

    constexpr std::array<Scalar, 8> weights_4 = {-3.384488523e-01, 8.180598728e-03, -1.237603743e-02, -6.497446299e-01, 2.065965533e-01, 2.091013044e-01, 4.104950726e-01, -6.879119277e-01};
    constexpr std::array<Scalar, 8> biases_4 = {-6.868880155e-05, 1.835801377e-04, 9.248071583e-04, -1.925037213e-04, 4.759973963e-04, -2.119484125e-04, -7.063761004e-04, -7.829110837e-05};

    constexpr std::array<Scalar, 8> gamma_5 = {9.977204204e-01, 1.003378391e+00, 9.875956178e-01, 1.001788735e+00, 1.006754994e+00, 9.956783056e-01, 9.972378612e-01, 1.001842976e+00};
    constexpr std::array<Scalar, 8> beta_5 = {-3.261103760e-03, 5.636610091e-03, -1.256635785e-02, 3.614827758e-03, -1.137817185e-02, -8.364913054e-03, -2.582596848e-03, -5.586442538e-03};
    constexpr std::array<Scalar, 8> mean_5 = {-1.891811378e-02, 4.054055025e-04, -1.005842932e-03, -3.627588972e-02, 1.136499736e-02, 1.175613515e-02, 2.291415259e-02, -3.838510439e-02};
    constexpr std::array<Scalar, 8> variance_5 = {8.661125302e-01, 8.600608706e-01, 8.600775599e-01, 8.823249936e-01, 8.622338772e-01, 8.624158502e-01, 8.689885736e-01, 8.849931955e-01};
    constexpr Scalar epsilon_5 = 1.000000000e-03;

    // Dummy placeholders for weights_7 and biases_7:
    constexpr std::array<Scalar, 9> weights_7 = {0};
    constexpr std::array<Scalar, 9> biases_7 = {0};

    constexpr std::array<Scalar, 16> gamma_8 = {9.995368123e-01, 9.990187287e-01, 9.950011373e-01, 9.944267869e-01, 9.989464879e-01, 9.980170131e-01, 1.001037717e+00, 9.916695356e-01, 1.007386088e+00, 1.001566410e+00, 9.964578748e-01, 1.003946424e+00, 9.937018752e-01, 1.004101038e+00, 9.985683560e-01, 9.951828718e-01};
    constexpr std::array<Scalar, 16> beta_8 = {-4.312225617e-03, 5.308859050e-03, 3.266467247e-03, -6.956706755e-03, -7.900752244e-04, -9.034545161e-03, -4.551545251e-03, 3.614332527e-03, 8.326089010e-03, 9.774068370e-03, -1.104702987e-02, -4.147845320e-03, -1.053798478e-02, 3.656426910e-03, -6.424990017e-03, -8.039580658e-03};
    constexpr std::array<Scalar, 16> mean_8 = {1.055789553e-02, -2.713145223e-03, -1.976119913e-02, 9.343987331e-03, -7.575007621e-03, -1.265626680e-02, -7.342638448e-03, 5.059114192e-03, 1.869842596e-02, 2.503054403e-02, 1.544746570e-02, 5.444306880e-03, 1.837889478e-02, -2.081481740e-02, 1.448171306e-02, 2.686110325e-02};
    constexpr std::array<Scalar, 16> variance_8 = {8.612734079e-01, 8.654001951e-01, 8.618488312e-01, 8.625833988e-01, 8.634274006e-01, 8.619918823e-01, 8.674398661e-01, 8.626600504e-01, 8.671016693e-01, 8.649455905e-01, 8.660261631e-01, 8.651937246e-01, 8.629028797e-01, 8.625113964e-01, 8.626952767e-01, 8.721811771e-01};
    constexpr Scalar epsilon_8 = 1.000000000e-03;

    // Dummy placeholders for weights_10 and biases_10:
    constexpr std::array<Scalar, 16> weights_10 = {0};
    constexpr std::array<Scalar, 16> biases_10 = {0};

    constexpr std::array<Scalar, 16> gamma_11 = {9.893945456e-01, 9.936457276e-01, 9.940803051e-01, 9.898499846e-01, 9.969305992e-01, 1.012331128e+00, 9.913190603e-01, 1.006539583e+00, 9.865819216e-01, 9.871383309e-01, 1.003752947e+00, 1.003496289e+00, 9.867479205e-01, 9.967375994e-01, 9.860343933e-01, 9.937593341e-01};
    constexpr std::array<Scalar, 16> beta_11 = {-9.882653132e-03, -6.501554046e-03, -5.548204295e-03, -9.637972340e-03, -2.439140342e-03, 1.209408790e-02, -8.228344843e-03, 5.780874752e-03, -1.316933148e-02, -1.294082310e-02, 2.341107465e-03, 3.389762016e-03, -1.341732219e-02, -2.059475984e-03, -1.393211633e-02, -6.092781201e-03};
    constexpr std::array<Scalar, 16> mean_11 = {1.730349148e-03, -1.611777022e-02, -7.250561845e-03, 2.701438032e-02, -1.652650535e-02, 1.228309050e-02, 1.864222926e-03, -4.072531126e-03, 2.194871567e-02, 7.743681781e-04, 2.177114785e-02, -2.254217304e-02, -1.188866049e-02, 1.673878543e-02, -1.164209004e-02, 1.289699576e-03};
    constexpr std::array<Scalar, 16> variance_11 = {8.633658886e-01, 8.629000187e-01, 8.651255965e-01, 8.650976419e-01, 8.645109534e-01, 8.669048548e-01, 8.641960621e-01, 8.616101742e-01, 8.660686612e-01, 8.655049801e-01, 8.661211133e-01, 8.645362258e-01, 8.641509414e-01, 8.652913570e-01, 8.637973070e-01, 8.635177016e-01};
    constexpr Scalar epsilon_11 = 1.000000000e-03;

    constexpr std::array<Scalar, 80> weights_14 = {-2.593242824e-01, -1.817701617e-03, -2.243622094e-01, 5.540617183e-02, -3.547773659e-01, 2.232497931e-01, 4.942587912e-01, 2.675659060e-01, -2.854668796e-01, 4.675116539e-01, -1.242927834e-01, 6.819196511e-03, -1.655491889e-01, 2.965084910e-01, 9.190067649e-02, 2.348518223e-01, 4.120061994e-01, 2.381704301e-01, -2.898204625e-01, -3.152502477e-01, -3.612616360e-01, -1.551427841e-01, -3.450012207e-01, 4.535778463e-01, -1.745133996e-01, 3.493096232e-01, -4.591509402e-01, 1.801212430e-01, -2.207425237e-02, -6.352915615e-02, 3.330100179e-01, 3.624832332e-01, 2.281507254e-01, 2.383794188e-01, 1.793605536e-01, 6.489153951e-02, -7.115987688e-02, 4.436414540e-01, 8.742903173e-02, 4.160164893e-01, -2.457726002e-01, 4.924044311e-01, -1.796402633e-01, -1.882847846e-01, -4.156348705e-01, 8.175920695e-02, 4.515710473e-01, 4.605799615e-01, 4.179961681e-01, 1.939453036e-01, -3.019922674e-01, -2.795309126e-01, 3.146932125e-01, 1.130922660e-01, 3.288639784e-01, 2.362502962e-01, 2.809759229e-02, -4.288273156e-01, -4.865388870e-01, 1.458462561e-03, -3.914014995e-01, 4.421942234e-01, 9.422065318e-02, -1.021793783e-01, -5.545492843e-02, -1.794041395e-01, -7.232995331e-02, -4.631054103e-01, 4.324167967e-01, 6.820437312e-02, -2.545875311e-02, 3.860474825e-01, 2.318956703e-01, -2.549740952e-03, -1.874455959e-01, -4.551549554e-01, -9.686171263e-02, 2.129567266e-01, 3.922936916e-01, -2.017081082e-01};
    constexpr std::array<Scalar, 5> biases_14 = {1.124452706e-02, -1.256507356e-02, -1.260973746e-03, 1.542300452e-03, 3.139118198e-03};

    //
    // Inline lambda activation functions
    //
    auto linear = linear_lambda;
    auto relu = relu_lambda;
    auto sigmoid = sigmoid_lambda;
    auto tanhCustom = tanhCustom_lambda;

    //
    // Processing layers
    //

    // Layer 1: Depthwise Convolution using weights_1 & biases_1
    // Assumed input dimensions: in_channels = 1, in_height = 8, in_width = 8, kernel 3x3, stride 1, pad 1.
    constexpr int in_channels = 1, in_height = 8, in_width = 8, kernel_h = 3, kernel_w = 3, stride_h = 1, stride_w = 1, pad_h = 1, pad_w = 1;
    // For simplicity, assume output size equals input size.
    std::array<Scalar, 8 * 8 * 1> depthwise_output;
    conv2DForward<Scalar, 1, 8, 8>(
        depthwise_output.data(), model_input.data(), weights_1.data(), biases_1.data(),
        in_channels, in_height, in_width, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
        linear, 0.0);
    log_intermediate_values(std::vector<Scalar>(depthwise_output.begin(), depthwise_output.end()));

    std::array<Scalar, 1> layer_2_output;
    batchNormalization<Scalar, 1>(layer_2_output.data(), depthwise_output.data(), gamma_2.data(), beta_2.data(), mean_2.data(), variance_2.data(), epsilon_2);
    log_intermediate_values(std::vector<Scalar>(layer_2_output.begin(), layer_2_output.end()));

    std::array<Scalar, 1> layer_3_output;
    linear(layer_3_output[0], layer_2_output[0], 0.0);
    log_intermediate_values(std::vector<Scalar>(layer_3_output.begin(), layer_3_output.end()));

    // Layer 4: Standard 2D Convolution
    constexpr int in_height_4 = 8, in_width_4 = 8, in_channels_4 = 1, kernel_h_4 = 1, kernel_w_4 = 1, stride_h_4 = 1, stride_w_4 = 1, pad_h_4 = 0, pad_w_4 = 0;
    constexpr int out_height_4 = (in_height_4 + 2 * pad_h_4 - kernel_h_4) / stride_h_4 + 1;
    constexpr int out_width_4 = (in_width_4 + 2 * pad_w_4 - kernel_w_4) / stride_w_4 + 1;
    std::array<Scalar, out_height_4 * out_width_4 * 8> layer_4_output;
    conv2DForward<Scalar, 8, out_height_4, out_width_4>(
        layer_4_output.data(), layer_3_output.data(), weights_4.data(), biases_4.data(),
        in_channels_4, in_height_4, in_width_4, kernel_h_4, kernel_w_4, stride_h_4, stride_w_4, pad_h_4, pad_w_4,
        linear, 0.0);
    log_intermediate_values(std::vector<Scalar>(layer_4_output.begin(), layer_4_output.end()));

    std::array<Scalar, 8> layer_5_output;
    batchNormalization<Scalar, 8>(layer_5_output.data(), layer_4_output.data(), gamma_5.data(), beta_5.data(), mean_5.data(), variance_5.data(), epsilon_5);
    log_intermediate_values(std::vector<Scalar>(layer_5_output.begin(), layer_5_output.end()));

    std::array<Scalar, 8> layer_6_output;
    for (int i = 0; i < 8; i++) {
        linear(layer_6_output[i], layer_5_output[i], 0.0);
    }
    log_intermediate_values(std::vector<Scalar>(layer_6_output.begin(), layer_6_output.end()));

    // Layer 7: Separable Convolution using dummy weights_7 & biases_7
    constexpr int in_channels_7 = 1, in_height_7 = 8, in_width_7 = 8, kernel_h_7 = 3, kernel_w_7 = 3, stride_h_7 = 1, stride_w_7 = 1, pad_h_7 = 1, pad_w_7 = 1;
    constexpr int out_height_7 = (in_height_7 + 2 * pad_h_7 - kernel_h_7) / stride_h_7 + 1;
    constexpr int out_width_7 = (in_width_7 + 2 * pad_w_7 - kernel_w_7) / stride_w_7 + 1;
    std::array<Scalar, out_height_7 * out_width_7 * 16> layer_7_output;
    separableConv2DForward<Scalar, 16, out_height_7, out_width_7, in_channels_7, in_height_7, in_width_7>(
        layer_7_output.data(), layer_6_output.data(), 
        weights_7.data(), weights_7.data(), biases_7.data(),
        kernel_h_7, kernel_w_7, stride_h_7, stride_w_7, pad_h_7, pad_w_7,
        linear, 0.0);
    log_intermediate_values(std::vector<Scalar>(layer_7_output.begin(), layer_7_output.end()));

    std::array<Scalar, 16> layer_8_output;
    batchNormalization<Scalar, 16>(layer_8_output.data(), layer_7_output.data(), gamma_8.data(), beta_8.data(), mean_8.data(), variance_8.data(), epsilon_8);
    log_intermediate_values(std::vector<Scalar>(layer_8_output.begin(), layer_8_output.end()));

    std::array<Scalar, 16> layer_9_output;
    for (int i = 0; i < 16; i++) {
        linear(layer_9_output[i], layer_8_output[i], 0.0);
    }
    log_intermediate_values(std::vector<Scalar>(layer_9_output.begin(), layer_9_output.end()));

    // Layer 10: Separable Convolution using dummy weights_10 & biases_10
    constexpr int in_channels_10 = 1, in_height_10 = 8, in_width_10 = 8, kernel_h_10 = 3, kernel_w_10 = 3, stride_h_10 = 1, stride_w_10 = 1, pad_h_10 = 1, pad_w_10 = 1;
    constexpr int out_height_10 = (in_height_10 + 2 * pad_h_10 - kernel_h_10) / stride_h_10 + 1;
    constexpr int out_width_10 = (in_width_10 + 2 * pad_w_10 - kernel_w_10) / stride_w_10 + 1;
    std::array<Scalar, out_height_10 * out_width_10 * 16> layer_10_output;
    separableConv2DForward<Scalar, 16, out_height_10, out_width_10, in_channels_10, in_height_10, in_width_10>(
        layer_10_output.data(), layer_9_output.data(),
        weights_10.data(), weights_10.data(), biases_10.data(),
        kernel_h_10, kernel_w_10, stride_h_10, stride_w_10, pad_h_10, pad_w_10,
        linear, 0.0);
    log_intermediate_values(std::vector<Scalar>(layer_10_output.begin(), layer_10_output.end()));

    std::array<Scalar, 16> layer_11_output;
    batchNormalization<Scalar, 16>(layer_11_output.data(), layer_10_output.data(), gamma_11.data(), beta_11.data(), mean_11.data(), variance_11.data(), epsilon_11);
    log_intermediate_values(std::vector<Scalar>(layer_11_output.begin(), layer_11_output.end()));

    std::array<Scalar, 16> layer_12_output;
    for (int i = 0; i < 16; i++) {
        linear(layer_12_output[i], layer_11_output[i], 0.0);
    }
    log_intermediate_values(std::vector<Scalar>(layer_12_output.begin(), layer_12_output.end()));

    std::array<Scalar, 16> layer_13_output;
    for (int i = 0; i < 16; i++) {
        linear(layer_13_output[i], layer_12_output[i], 0.0);
    }
    log_intermediate_values(std::vector<Scalar>(layer_13_output.begin(), layer_13_output.end()));

    std::array<Scalar, 5> layer_14_output;
    forwardPass<Scalar, 5>(layer_14_output.data(), layer_13_output.data(), weights_14.data(), biases_14.data(), 16, linear, 0.0);
    log_intermediate_values(std::vector<Scalar>(layer_14_output.begin(), layer_14_output.end()));

    auto model_output = layer_14_output;
    return model_output;
}

// Function to get intermediate values for debugging
template <typename Scalar>
std::vector<std::vector<Scalar>> get_intermediate_values() {
    return intermediate_values<Scalar>;
}
