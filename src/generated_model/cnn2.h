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
auto cnn2(const std::array<Scalar, 28>& initial_input) { 

    std::array<Scalar, 28> model_input = initial_input;

    if (model_input.size() != 28) { throw std::invalid_argument("Invalid input size. Expected size: 28"); }

    //\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\// 

    constexpr std::array<Scalar, 9> weights_1 = {-5.349085927e-01, -3.953629136e-01, 5.769196749e-01, 2.306873202e-01, -3.479735255e-01, 2.409154177e-02, -7.580578327e-03, -1.298219860e-01, 3.050568700e-01};

    constexpr std::array<Scalar, 1> biases_1 = {0.000000000e+00};

    constexpr std::array<Scalar, 1> gamma_2 = {1.000000000e+00};

    constexpr std::array<Scalar, 1> beta_2 = {0.000000000e+00};

    constexpr std::array<Scalar, 1> mean_2 = {0.000000000e+00};

    constexpr std::array<Scalar, 1> variance_2 = {1.000000000e+00};

    constexpr Scalar epsilon_2 = 1.000000000e-03;

    constexpr std::array<Scalar, 8> weights_4 = {-7.602812052e-01, -1.702624559e-01, -3.691533208e-01, 3.981792927e-01, -6.285692453e-01, -3.023523092e-02, -4.737848341e-01, -1.993902326e-01};

    constexpr std::array<Scalar, 8> biases_4 = {0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00};

    constexpr std::array<Scalar, 8> gamma_5 = {1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00};

    constexpr std::array<Scalar, 8> beta_5 = {0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00};

    constexpr std::array<Scalar, 8> mean_5 = {0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00};

    constexpr std::array<Scalar, 8> variance_5 = {1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00};

    constexpr Scalar epsilon_5 = 1.000000000e-03;

    constexpr std::array<Scalar, 16> gamma_8 = {1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00};

    constexpr std::array<Scalar, 16> beta_8 = {0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00};

    constexpr std::array<Scalar, 16> mean_8 = {0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00};

    constexpr std::array<Scalar, 16> variance_8 = {1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00};

    constexpr Scalar epsilon_8 = 1.000000000e-03;

    constexpr std::array<Scalar, 16> gamma_11 = {1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00};

    constexpr std::array<Scalar, 16> beta_11 = {0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00};

    constexpr std::array<Scalar, 16> mean_11 = {0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00};

    constexpr std::array<Scalar, 16> variance_11 = {1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00};

    constexpr Scalar epsilon_11 = 1.000000000e-03;

    constexpr std::array<Scalar, 160> weights_14 = {-4.538519681e-01, -4.719312787e-01, 4.234404564e-01, -1.572263241e-01, 1.974256635e-01, -6.457334757e-02, -4.587680399e-01, 1.268955469e-01, -2.214686871e-01, -1.959682405e-01, 2.877238393e-01, -1.922371984e-01, 5.219787359e-02, -4.270098805e-01, 2.207436562e-01, 3.585450649e-01, 4.003602266e-02, -4.615793526e-01, -3.933028281e-01, -1.264771819e-02, 1.762896776e-01, -4.196578264e-02, 2.755047083e-01, 2.737072110e-01, -1.836961806e-01, -3.717162907e-01, 3.120126128e-01, -7.735794783e-02, 4.097400904e-01, -2.207590342e-01, 1.740936637e-01, 3.874889016e-01, -1.465497017e-01, -1.531590521e-01, 3.294098973e-01, 1.527073979e-02, -3.574962616e-01, 2.804372907e-01, 3.728836775e-02, 2.826687098e-01, 3.342210650e-01, -5.636778474e-02, 1.532015204e-01, 7.394003868e-02, 1.167185307e-01, 1.338214874e-01, 1.428371072e-01, -3.053908348e-01, 3.712363839e-01, 2.579778433e-02, 2.762541771e-01, 2.561725974e-01, -3.214367032e-01, -4.797858000e-01, -2.635575533e-01, -4.328543544e-01, -2.483407706e-01, -1.534689665e-01, 2.418982983e-02, 2.396452427e-01, 2.798523307e-01, -4.796395600e-01, 1.185452342e-01, -2.635926008e-01, -2.972893715e-01, -1.996828616e-01, 1.161691546e-02, 4.616597891e-01, 4.666513205e-02, -3.114143014e-01, 4.251887202e-01, 4.551172256e-02, 1.644828916e-01, 3.127519488e-01, 4.588077664e-01, -1.980923414e-01, 1.469019055e-02, 2.694371939e-01, -8.348050714e-02, -3.516963124e-02, 2.605209947e-01, -4.138149321e-01, 3.019044399e-01, 1.163782477e-01, -2.146768868e-01, 1.311496496e-01, -2.655094266e-01, 4.511533380e-01, 1.704335213e-01, 2.037199140e-01, -2.068467736e-01, -4.468000829e-01, 4.591721296e-01, -1.815799475e-01, 3.688514829e-01, -3.364924788e-01, 2.640116811e-01, 3.381446004e-01, 2.988597155e-01, 2.010535598e-01, -1.728549898e-01, -2.398229837e-01, -1.208187044e-01, -3.214454353e-01, -2.611737847e-01, 3.020291924e-01, 3.563548326e-01, 2.894985080e-01, -3.709557056e-01, 2.893975973e-01, -4.183217883e-01, 2.978304029e-01, 4.016893506e-01, -3.722172678e-01, -3.911215663e-01, 4.090535641e-01, 3.692655563e-01, -1.621398926e-01, -7.442396879e-02, -4.000914097e-01, 2.627351284e-01, -4.723225236e-01, -2.449114323e-01, 2.028893232e-01, -1.434634030e-01, -3.624884188e-01, -2.360962033e-01, -2.853773832e-01, -2.468513846e-01, -1.527991891e-02, 1.077081561e-01, -1.522198617e-01, 1.576837897e-01, -1.346381009e-01, 2.746360898e-01, 3.752088547e-01, -3.417281210e-01, -4.429389536e-01, -1.093549132e-01, 2.151333094e-01, -2.253925502e-01, -8.993029594e-02, 9.943318367e-02, 4.485761523e-01, 1.044525504e-01, 4.159059525e-01, 4.362508059e-01, -2.399726808e-01, 8.897119761e-02, -3.470993638e-01, 2.975446582e-01, -3.391267359e-01, 4.200862646e-01, -1.717329025e-01, -3.962734640e-01, -1.146138906e-01, -2.597278357e-02, 3.656512499e-02, -3.285474777e-01, 2.502011061e-01};

    constexpr std::array<Scalar, 10> biases_14 = {0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00};

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

    std::array<Scalar, 10> layer_14_output;
    forwardPass<Scalar, 10>(layer_14_output.data(), layer_13_output.data(), weights_14.data(), biases_14.data(), 16, linear, 0.0);

    std::array<Scalar, 10> model_output = layer_14_output;

    return model_output;
}
