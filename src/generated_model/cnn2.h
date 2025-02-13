#pragma once
#include <iostream>
#include <array>
#include <random>
#include <cmath>
#include <functional>

template<typename Scalar>
using activationFunction = void(*)(Scalar&, Scalar, Scalar);

//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//

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

template<typename Scalar, int out_channels, int out_height, int out_width>
void depthwiseConv2DForward(Scalar* outputs, const Scalar* inputs, const Scalar* weights, const Scalar* biases,
                            int in_channels, int in_height, int in_width,
                            int kernel_h, int kernel_w, int stride_h, int stride_w,
                            int pad_h, int pad_w,
                            activationFunction<Scalar> activation_function, Scalar alpha) noexcept {
    // Simplified depthwise convolution implementation (each input channel is convolved independently)
    for (int c = 0; c < in_channels; ++c) {
        for (int oh = 0; oh < out_height; ++oh) {
            for (int ow = 0; ow < out_width; ++ow) {
                Scalar sum = 0;
                for (int kh = 0; kh < kernel_h; ++kh) {
                    for (int kw = 0; kw < kernel_w; ++kw) {
                        int in_h = oh * stride_h - pad_h + kh;
                        int in_w = ow * stride_w - pad_w + kw;
                        if(in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width){
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

template<typename Scalar, int out_channels, int out_height, int out_width>
void separableConv2DForward(Scalar* outputs, const Scalar* inputs, const Scalar* depthwise_weights, const Scalar* pointwise_weights, const Scalar* biases,
                            int in_channels, int in_height, int in_width,
                            int kernel_h, int kernel_w, int stride_h, int stride_w,
                            int pad_h, int pad_w,
                            activationFunction<Scalar> activation_function, Scalar alpha) noexcept {
    // First perform depthwise convolution (this is a simplified approach)
    const int depthwise_output_size = in_height * in_width * in_channels; // assuming same spatial dims for simplicity
    Scalar depthwise_output[depthwise_output_size];
    depthwiseConv2DForward<Scalar, in_channels, in_height, in_width>(depthwise_output, inputs, depthwise_weights, biases, in_channels, in_height, in_width, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, linear, 0.0);
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

template <typename Scalar = double>
auto cnn2(const std::array<Scalar, 8>& initial_input) { 

    std::array<Scalar, 8> model_input = initial_input;

    if (model_input.size() != 8) { throw std::invalid_argument("Invalid input size. Expected size: 8"); }

    //\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\// 

    constexpr std::array<Scalar, 9> weights_1 = {-1.649156809e-01, 5.690510273e-01, -4.962989092e-01, 1.706098020e-01, -2.220861316e-01, -1.560654491e-01, 4.322054684e-01, -3.612734377e-01, -3.583632112e-01};

    constexpr std::array<Scalar, 1> biases_1 = {-6.131598639e-05};

    constexpr std::array<Scalar, 1> gamma_2 = {9.994760156e-01};

    constexpr std::array<Scalar, 1> beta_2 = {-6.030072924e-03};

    constexpr std::array<Scalar, 1> mean_2 = {-3.249455616e-02};

    constexpr std::array<Scalar, 1> variance_2 = {8.755799532e-01};

    constexpr Scalar epsilon_2 = 1.000000000e-03;

    constexpr std::array<Scalar, 8> weights_4 = {-1.640573889e-01, 3.321343660e-01, 5.549435019e-01, -4.922070503e-01, 7.163643837e-01, 6.127314568e-01, -5.715216696e-02, 5.852361917e-01};

    constexpr std::array<Scalar, 8> biases_4 = {-1.152500918e-04, 5.487648887e-04, 4.078895290e-05, 1.353159678e-05, 5.827425412e-05, 1.509300273e-05, -2.553075901e-04, 1.399490720e-04};

    constexpr std::array<Scalar, 8> gamma_5 = {1.005890012e+00, 9.880468249e-01, 9.993941188e-01, 1.000773430e+00, 9.967370033e-01, 9.954102039e-01, 9.986677170e-01, 1.010096908e+00};

    constexpr std::array<Scalar, 8> beta_5 = {6.066386588e-03, -1.248240843e-02, -9.623068385e-03, 1.471803873e-03, 3.133545397e-05, -8.483674377e-03, -3.456272534e-04, 1.820692793e-03};

    constexpr std::array<Scalar, 8> mean_5 = {-9.010137059e-03, 1.889693923e-02, 3.097571991e-02, -2.742240578e-02, 4.004621506e-02, 3.428971395e-02, -3.193665529e-03, 3.240194544e-02};

    constexpr std::array<Scalar, 8> variance_5 = {8.613024354e-01, 8.655096292e-01, 8.747722507e-01, 8.715911508e-01, 8.846498728e-01, 8.780816197e-01, 8.602144718e-01, 8.761454821e-01};

    constexpr Scalar epsilon_5 = 1.000000000e-03;

    constexpr std::array<Scalar, 16> gamma_8 = {9.925953150e-01, 1.004570842e+00, 9.998089671e-01, 9.974120855e-01, 1.003188729e+00, 1.005252242e+00, 9.984737039e-01, 9.937770367e-01, 1.006073833e+00, 9.982897043e-01, 9.985503554e-01, 9.997153282e-01, 9.963559508e-01, 9.976913929e-01, 9.972578287e-01, 9.924487472e-01};

    constexpr std::array<Scalar, 16> beta_8 = {-9.894247167e-03, -1.048195176e-02, 5.096609821e-04, 4.387978697e-04, -9.800216183e-03, 6.980425678e-03, -4.002131522e-03, -1.318610180e-02, -1.248841058e-03, 3.594744718e-03, 1.001322269e-02, -8.909082040e-03, -1.120249159e-03, -3.949348815e-03, 9.461207315e-03, -7.159914356e-03};

    constexpr std::array<Scalar, 16> mean_8 = {-1.020016801e-02, 2.452135086e-02, 2.015231177e-02, 1.837279648e-02, -5.266285036e-03, -5.877926014e-03, 2.712199092e-02, -4.714940675e-03, -1.451544580e-03, 1.481223758e-02, -5.703898612e-03, 1.718694577e-03, -2.060205303e-02, 7.105038501e-03, -1.538831129e-04, -1.120857056e-02};

    constexpr std::array<Scalar, 16> variance_8 = {8.620697856e-01, 8.720425963e-01, 8.645434380e-01, 8.754804730e-01, 8.690392971e-01, 8.653692603e-01, 8.734630942e-01, 8.650845289e-01, 8.664746881e-01, 8.723049760e-01, 8.625785112e-01, 8.696178198e-01, 8.694362044e-01, 8.631252646e-01, 8.691535592e-01, 8.615903258e-01};

    constexpr Scalar epsilon_8 = 1.000000000e-03;

    constexpr std::array<Scalar, 16> gamma_11 = {9.896695018e-01, 9.883487821e-01, 9.916002750e-01, 9.897872210e-01, 9.915282130e-01, 1.008236885e+00, 1.010128379e+00, 1.004228711e+00, 9.985761642e-01, 1.009705186e+00, 1.008660078e+00, 1.002184749e+00, 1.009857774e+00, 9.927137494e-01, 9.936082363e-01, 9.913144708e-01};

    constexpr std::array<Scalar, 16> beta_11 = {-1.080155186e-02, -1.151598338e-02, -9.425890632e-03, -9.141894989e-03, -8.683339693e-03, 8.822284639e-03, 9.976616129e-03, 4.476278555e-03, -4.477286129e-04, 1.045026351e-02, 9.887434542e-03, 1.383902272e-03, 1.003238931e-02, -7.555453107e-03, -6.825971883e-03, -8.962598629e-03};

    constexpr std::array<Scalar, 16> mean_11 = {7.069018669e-03, 1.886645518e-02, -6.048891228e-04, -1.943129301e-02, -1.543682627e-02, 3.285927000e-03, 5.058856681e-03, -7.361865602e-03, 1.347815269e-03, -1.771015674e-02, 1.604300737e-02, 1.227953192e-02, -4.137739539e-02, -2.111596242e-02, 1.647539064e-02, -5.943099502e-03};

    constexpr std::array<Scalar, 16> variance_11 = {8.637157083e-01, 8.666414022e-01, 8.641531467e-01, 8.645910621e-01, 8.667618632e-01, 8.677000403e-01, 8.697998524e-01, 8.682133555e-01, 8.644905090e-01, 8.675518036e-01, 8.663613200e-01, 8.626252413e-01, 8.682324886e-01, 8.731962442e-01, 8.682899475e-01, 8.630052209e-01};

    constexpr Scalar epsilon_11 = 1.000000000e-03;

    constexpr std::array<Scalar, 80> weights_14 = {-2.776539922e-01, -1.364495456e-01, 5.124819875e-01, 2.969165742e-01, -4.761323929e-01, -2.527607679e-01, 3.066371381e-01, 4.941866696e-01, 2.852915525e-01, -5.098662376e-01, -1.404468529e-02, 2.918707207e-02, -2.510680258e-02, 4.487560987e-01, -2.949966192e-01, 1.955969036e-01, -5.823041499e-02, 5.182762742e-01, -3.895570636e-01, -3.140293956e-01, -4.979765117e-01, -3.999322057e-01, -8.240258694e-02, 1.376773119e-01, 7.693820447e-02, 3.282937706e-01, 5.017834902e-01, 8.676926792e-02, 3.057035208e-01, 2.820709944e-01, 8.838944882e-02, -2.304034680e-01, -8.029853925e-03, -2.795876563e-01, 1.728553772e-01, -2.464171499e-01, 2.954655290e-01, -5.457244813e-02, -3.018944263e-01, -1.134289131e-01, -6.917168945e-02, 3.750356436e-01, 5.011566877e-01, -6.122926250e-02, 2.249047756e-01, -2.514322400e-01, -2.515777051e-01, -5.116434693e-01, 2.181539536e-01, 3.369432986e-01, 2.113310695e-01, 1.441702992e-01, 1.086386144e-01, -1.298988611e-01, 4.420351982e-01, 2.406175584e-01, -1.488855332e-01, -4.624611437e-01, -5.189665779e-02, -4.323709011e-01, -1.048661917e-01, 1.106081456e-01, -3.120171130e-01, -1.669516563e-01, 2.097222060e-01, 4.839618802e-01, -2.045327276e-01, 1.951551139e-01, 2.995562553e-01, -3.760465384e-01, 1.518299878e-01, -1.356608048e-02, 3.470164537e-02, 3.320226371e-01, -4.551234469e-02, -5.258958414e-02, 1.524429172e-01, -1.262062490e-01, 3.215595484e-01, -3.425247073e-01};

    constexpr std::array<Scalar, 5> biases_14 = {1.010765880e-02, 4.955140408e-03, -1.091245282e-02, -8.223558776e-03, 1.020564791e-02};

    //\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//

    auto linear = [](Scalar& output, Scalar input, Scalar alpha) noexcept {
        output = input;
    };

    //\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\// 

    // NEW CODE: Convolution layer processing for layer 1
    // TODO: Specify input dimensions for convolution layer 1
    constexpr int in_height_1 = /* input height */ 0;
    constexpr int in_width_1 = /* input width */ 0;
    constexpr int in_channels_1 = /* input channels */ 0;
    constexpr int kernel_h_1 = 3;
    constexpr int kernel_w_1 = 3;
    constexpr int stride_h_1 = 1;
    constexpr int stride_w_1 = 1;
    constexpr int pad_h_1 = 1;
    constexpr int pad_w_1 = 1;
    constexpr int out_height_1 = (in_height_1 + 2 * pad_h_1 - kernel_h_1) / stride_h_1 + 1;
    constexpr int out_width_1 = (in_width_1 + 2 * pad_w_1 - kernel_w_1) / stride_w_1 + 1;
    std::array<Scalar, out_height_1 * out_width_1 * 1> layer_1_output;
    depthwiseConv2DForward<Scalar, 1, out_height_1, out_width_1>(layer_1_output.data(), model_input.data(), weights_1.data(), biases_1.data(), in_channels_1, in_height_1, in_width_1, kernel_h_1, kernel_w_1, stride_h_1, stride_w_1, pad_h_1, pad_w_1, linear, 0.0);

    std::array<Scalar, 1> layer_2_output;
    batchNormalization<Scalar, 1>(layer_2_output.data(), layer_1_output.data(), gamma_2.data(), beta_2.data(), mean_2.data(), variance_2.data(), epsilon_2);

    std::array<Scalar, 1> layer_3_output;
    linear(layer_3_output[0], layer_2_output[0], 0.0);

    // NEW CODE: Convolution layer processing for layer 4
    // TODO: Specify input dimensions for convolution layer 4
    constexpr int in_height_4 = /* input height */ 0;
    constexpr int in_width_4 = /* input width */ 0;
    constexpr int in_channels_4 = /* input channels */ 0;
    constexpr int kernel_h_4 = 1;
    constexpr int kernel_w_4 = 1;
    constexpr int stride_h_4 = 1;
    constexpr int stride_w_4 = 1;
    constexpr int pad_h_4 = 0;
    constexpr int pad_w_4 = 0;
    constexpr int out_height_4 = (in_height_4 + 2 * pad_h_4 - kernel_h_4) / stride_h_4 + 1;
    constexpr int out_width_4 = (in_width_4 + 2 * pad_w_4 - kernel_w_4) / stride_w_4 + 1;
    std::array<Scalar, out_height_4 * out_width_4 * 8> layer_4_output;
    conv2DForward<Scalar, 8, out_height_4, out_width_4>(layer_4_output.data(), layer_3_output.data(), weights_4.data(), biases_4.data(), in_channels_4, in_height_4, in_width_4, kernel_h_4, kernel_w_4, stride_h_4, stride_w_4, pad_h_4, pad_w_4, linear, 0.0);

    std::array<Scalar, 8> layer_5_output;
    batchNormalization<Scalar, 8>(layer_5_output.data(), layer_4_output.data(), gamma_5.data(), beta_5.data(), mean_5.data(), variance_5.data(), epsilon_5);

    std::array<Scalar, 8> layer_6_output;
    linear(layer_6_output[0], layer_5_output[0], 0.0);
    linear(layer_6_output[1], layer_5_output[1], 0.0);
    linear(layer_6_output[2], layer_5_output[2], 0.0);
    linear(layer_6_output[3], layer_5_output[3], 0.0);
    linear(layer_6_output[4], layer_5_output[4], 0.0);
    linear(layer_6_output[5], layer_5_output[5], 0.0);
    linear(layer_6_output[6], layer_5_output[6], 0.0);
    linear(layer_6_output[7], layer_5_output[7], 0.0);

    // NEW CODE: Convolution layer processing for layer 7
    // TODO: Specify input dimensions for convolution layer 7
    constexpr int in_height_7 = /* input height */ 0;
    constexpr int in_width_7 = /* input width */ 0;
    constexpr int in_channels_7 = /* input channels */ 0;
    constexpr int kernel_h_7 = 3;
    constexpr int kernel_w_7 = 3;
    constexpr int stride_h_7 = 1;
    constexpr int stride_w_7 = 1;
    constexpr int pad_h_7 = 1;
    constexpr int pad_w_7 = 1;
    constexpr int out_height_7 = (in_height_7 + 2 * pad_h_7 - kernel_h_7) / stride_h_7 + 1;
    constexpr int out_width_7 = (in_width_7 + 2 * pad_w_7 - kernel_w_7) / stride_w_7 + 1;
    std::array<Scalar, out_height_7 * out_width_7 * 16> layer_7_output;
    separableConv2DForward<Scalar, 16, out_height_7, out_width_7>(layer_7_output.data(), layer_6_output.data(), weights_7.data(), biases_7.data(), in_channels_7, in_height_7, in_width_7, kernel_h_7, kernel_w_7, stride_h_7, stride_w_7, pad_h_7, pad_w_7, linear, 0.0);

    std::array<Scalar, 16> layer_8_output;
    batchNormalization<Scalar, 16>(layer_8_output.data(), layer_7_output.data(), gamma_8.data(), beta_8.data(), mean_8.data(), variance_8.data(), epsilon_8);

    std::array<Scalar, 16> layer_9_output;
    linear(layer_9_output[0], layer_8_output[0], 0.0);
    linear(layer_9_output[1], layer_8_output[1], 0.0);
    linear(layer_9_output[2], layer_8_output[2], 0.0);
    linear(layer_9_output[3], layer_8_output[3], 0.0);
    linear(layer_9_output[4], layer_8_output[4], 0.0);
    linear(layer_9_output[5], layer_8_output[5], 0.0);
    linear(layer_9_output[6], layer_8_output[6], 0.0);
    linear(layer_9_output[7], layer_8_output[7], 0.0);
    linear(layer_9_output[8], layer_8_output[8], 0.0);
    linear(layer_9_output[9], layer_8_output[9], 0.0);
    linear(layer_9_output[10], layer_8_output[10], 0.0);
    linear(layer_9_output[11], layer_8_output[11], 0.0);
    linear(layer_9_output[12], layer_8_output[12], 0.0);
    linear(layer_9_output[13], layer_8_output[13], 0.0);
    linear(layer_9_output[14], layer_8_output[14], 0.0);
    linear(layer_9_output[15], layer_8_output[15], 0.0);

    // NEW CODE: Convolution layer processing for layer 10
    // TODO: Specify input dimensions for convolution layer 10
    constexpr int in_height_10 = /* input height */ 0;
    constexpr int in_width_10 = /* input width */ 0;
    constexpr int in_channels_10 = /* input channels */ 0;
    constexpr int kernel_h_10 = 3;
    constexpr int kernel_w_10 = 3;
    constexpr int stride_h_10 = 1;
    constexpr int stride_w_10 = 1;
    constexpr int pad_h_10 = 1;
    constexpr int pad_w_10 = 1;
    constexpr int out_height_10 = (in_height_10 + 2 * pad_h_10 - kernel_h_10) / stride_h_10 + 1;
    constexpr int out_width_10 = (in_width_10 + 2 * pad_w_10 - kernel_w_10) / stride_w_10 + 1;
    std::array<Scalar, out_height_10 * out_width_10 * 16> layer_10_output;
    separableConv2DForward<Scalar, 16, out_height_10, out_width_10>(layer_10_output.data(), layer_9_output.data(), weights_10.data(), biases_10.data(), in_channels_10, in_height_10, in_width_10, kernel_h_10, kernel_w_10, stride_h_10, stride_w_10, pad_h_10, pad_w_10, linear, 0.0);

    std::array<Scalar, 16> layer_11_output;
    batchNormalization<Scalar, 16>(layer_11_output.data(), layer_10_output.data(), gamma_11.data(), beta_11.data(), mean_11.data(), variance_11.data(), epsilon_11);

    std::array<Scalar, 16> layer_12_output;
    linear(layer_12_output[0], layer_11_output[0], 0.0);
    linear(layer_12_output[1], layer_11_output[1], 0.0);
    linear(layer_12_output[2], layer_11_output[2], 0.0);
    linear(layer_12_output[3], layer_11_output[3], 0.0);
    linear(layer_12_output[4], layer_11_output[4], 0.0);
    linear(layer_12_output[5], layer_11_output[5], 0.0);
    linear(layer_12_output[6], layer_11_output[6], 0.0);
    linear(layer_12_output[7], layer_11_output[7], 0.0);
    linear(layer_12_output[8], layer_11_output[8], 0.0);
    linear(layer_12_output[9], layer_11_output[9], 0.0);
    linear(layer_12_output[10], layer_11_output[10], 0.0);
    linear(layer_12_output[11], layer_11_output[11], 0.0);
    linear(layer_12_output[12], layer_11_output[12], 0.0);
    linear(layer_12_output[13], layer_11_output[13], 0.0);
    linear(layer_12_output[14], layer_11_output[14], 0.0);
    linear(layer_12_output[15], layer_11_output[15], 0.0);

    std::array<Scalar, 16> layer_13_output;
    linear(layer_13_output[0], layer_12_output[0], 0.0);
    linear(layer_13_output[1], layer_12_output[1], 0.0);
    linear(layer_13_output[2], layer_12_output[2], 0.0);
    linear(layer_13_output[3], layer_12_output[3], 0.0);
    linear(layer_13_output[4], layer_12_output[4], 0.0);
    linear(layer_13_output[5], layer_12_output[5], 0.0);
    linear(layer_13_output[6], layer_12_output[6], 0.0);
    linear(layer_13_output[7], layer_12_output[7], 0.0);
    linear(layer_13_output[8], layer_12_output[8], 0.0);
    linear(layer_13_output[9], layer_12_output[9], 0.0);
    linear(layer_13_output[10], layer_12_output[10], 0.0);
    linear(layer_13_output[11], layer_12_output[11], 0.0);
    linear(layer_13_output[12], layer_12_output[12], 0.0);
    linear(layer_13_output[13], layer_12_output[13], 0.0);
    linear(layer_13_output[14], layer_12_output[14], 0.0);
    linear(layer_13_output[15], layer_12_output[15], 0.0);

    std::array<Scalar, 5> layer_14_output;
    forwardPass<Scalar, 5>(layer_14_output.data(), layer_13_output.data(), weights_14.data(), biases_14.data(), 16, linear, 0.0);

    std::array<Scalar, 5> model_output = layer_14_output;

    return model_output;
}
