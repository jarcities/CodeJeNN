#pragma once
#include <iostream>
#include <array>
#include <random>
#include <cmath>
#include <functional>
#include <stdexcept>

template<typename Scalar>
using activationFunction = void(*)(Scalar&, Scalar, Scalar);


//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// 


template<typename Scalar, int output_size>
void Dense(Scalar* outputs, const Scalar* inputs, const Scalar* weights, const Scalar* biases, int input_size, activationFunction<Scalar> activation_function, Scalar alpha) noexcept {
    for(int i = 0; i < output_size; ++i){
        Scalar sum = 0;
        for(int j = 0; j < input_size; ++j){
            sum += inputs[j] * weights[j * output_size + i];
        }
        sum += biases[i];
        activation_function(outputs[i], sum, alpha);
    }
}
    
template <typename Scalar, int out_size>
void conv1DForward(Scalar *outputs, const Scalar *inputs, const Scalar *weights, const Scalar *biases,
                   int in_size, int kernel_size, int stride, int pad,
                   activationFunction<Scalar> activation_function, Scalar alpha) noexcept
{
    for (int o = 0; o < out_size; ++o)
    {
        Scalar sum = 0;
        for (int k = 0; k < kernel_size; ++k)
        {
            int in_index = o * stride - pad + k;
            if (in_index >= 0 && in_index < in_size)
            {
                int weight_index = k * out_size + o;
                sum += inputs[in_index] * weights[weight_index];
            }
        }
        sum += biases[o];
        activation_function(outputs[o], sum, alpha);
    }
}

template <typename Scalar, int out_channels, int out_height, int out_width>
void conv2DForward(Scalar *outputs, const Scalar *inputs, const Scalar *weights, const Scalar *biases,
                   int in_channels, int in_height, int in_width,
                   int kernel_h, int kernel_w, int stride_h, int stride_w,
                   int pad_h, int pad_w,
                   activationFunction<Scalar> activation_function, Scalar alpha) noexcept
{
    for (int oc = 0; oc < out_channels; ++oc)
    {
        for (int oh = 0; oh < out_height; ++oh)
        {
            for (int ow = 0; ow < out_width; ++ow)
            {
                Scalar sum = 0;
                for (int ic = 0; ic < in_channels; ++ic)
                {
                    for (int kh = 0; kh < kernel_h; ++kh)
                    {
                        for (int kw = 0; kw < kernel_w; ++kw)
                        {
                            int in_h = oh * stride_h - pad_h + kh;
                            int in_w = ow * stride_w - pad_w + kw;
                            if (in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width)
                            {
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

template <typename Scalar, int out_channels, int out_depth, int out_height, int out_width>
void conv3DForward(Scalar *outputs, const Scalar *inputs, const Scalar *weights, const Scalar *biases,
                   int in_channels, int in_depth, int in_height, int in_width,
                   int kernel_d, int kernel_h, int kernel_w, int stride_d, int stride_h, int stride_w,
                   int pad_d, int pad_h, int pad_w,
                   activationFunction<Scalar> activation_function, Scalar alpha) noexcept
{
    // Simplified 3D convolution implementation
    for (int oc = 0; oc < out_channels; ++oc)
    {
        for (int od = 0; od < out_depth; ++od)
        {
            for (int oh = 0; oh < out_height; ++oh)
            {
                for (int ow = 0; ow < out_width; ++ow)
                {
                    Scalar sum = 0;
                    for (int ic = 0; ic < in_channels; ++ic)
                    {
                        for (int kd = 0; kd < kernel_d; ++kd)
                        {
                            for (int kh = 0; kh < kernel_h; ++kh)
                            {
                                for (int kw = 0; kw < kernel_w; ++kw)
                                {
                                    int in_d = od * stride_d - pad_d + kd;
                                    int in_h = oh * stride_h - pad_h + kh;
                                    int in_w = ow * stride_w - pad_w + kw;
                                    if (in_d >= 0 && in_d < in_depth &&
                                        in_h >= 0 && in_h < in_height &&
                                        in_w >= 0 && in_w < in_width)
                                    {
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

template <typename Scalar>
void depthwiseConv2DForward(Scalar *outputs, const Scalar *inputs, const Scalar *weights, const Scalar *biases,
                            int out_channels, int out_height, int out_width,
                            int in_channels, int in_height, int in_width,
                            int kernel_h, int kernel_w, int stride_h, int stride_w,
                            int pad_h, int pad_w,
                            activationFunction<Scalar> activation_function, Scalar alpha) noexcept
{
    for (int c = 0; c < in_channels; ++c)
    {
        for (int oh = 0; oh < out_height; ++oh)
        {
            for (int ow = 0; ow < out_width; ++ow)
            {
                Scalar sum = 0;
                for (int kh = 0; kh < kernel_h; ++kh)
                {
                    for (int kw = 0; kw < kernel_w; ++kw)
                    {
                        int in_h = oh * stride_h - pad_h + kh;
                        int in_w = ow * stride_w - pad_w + kw;
                        if (in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width)
                        {
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


template <typename Scalar>
void depthwiseForsSeparableConv2DForward(Scalar *outputs, const Scalar *inputs, const Scalar *weights, const Scalar *biases,
                            int out_channels, int out_height, int out_width,
                            int in_channels, int in_height, int in_width,
                            int kernel_h, int kernel_w, int stride_h, int stride_w,
                            int pad_h, int pad_w,
                            activationFunction<Scalar> activation_function, Scalar alpha) noexcept
{
    for (int c = 0; c < in_channels; ++c)
    {
        for (int oh = 0; oh < out_height; ++oh)
        {
            for (int ow = 0; ow < out_width; ++ow)
            {
                Scalar sum = 0;
                for (int kh = 0; kh < kernel_h; ++kh)
                {
                    for (int kw = 0; kw < kernel_w; ++kw)
                    {
                        int in_h = oh * stride_h - pad_h + kh;
                        int in_w = ow * stride_w - pad_w + kw;
                        if (in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width)
                        {
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

template <typename Scalar, int out_channels, int out_height, int out_width>
void separableConv2DForward(Scalar *outputs, const Scalar *inputs, const Scalar *depthwise_weights, const Scalar *pointwise_weights, const Scalar *biases,
                            int in_channels, int in_height, int in_width,
                            int kernel_h, int kernel_w, int stride_h, int stride_w,
                            int pad_h, int pad_w,
                            activationFunction<Scalar> activation_function, Scalar alpha) noexcept
{
    std::vector<Scalar> depthwise_output(in_height * in_width * in_channels, 0);
    std::vector<Scalar> zero_bias(in_channels, 0);
    depthwiseForsSeparableConv2DForward(
        depthwise_output.data(), inputs, depthwise_weights, zero_bias.data(), 
        in_channels, in_height, in_width,                                     
        in_channels, in_height, in_width,
        kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
        activation_function, alpha);

    for (int oc = 0; oc < out_channels; ++oc)
    {
        for (int i = 0; i < in_height * in_width; ++i)
        {
            Scalar sum = 0;
            for (int ic = 0; ic < in_channels; ++ic)
            {
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

template <typename Scalar>
void globalAvgPooling2D(Scalar *output, const Scalar *inputs, int in_height, int in_width, int channels) noexcept
{
    // Compute global average per channel.
    for (int c = 0; c < channels; ++c)
    {
        Scalar sum = 0;
        for (int h = 0; h < in_height; ++h)
        {
            for (int w = 0; w < in_width; ++w)
            {
                int idx = (h * in_width * channels) + (w * channels) + c;
                sum += inputs[idx];
            }
        }
        output[c] = sum / (in_height * in_width);
    }
}

template <typename Scalar, int channels, int height, int width>
void batchNormalization2D(Scalar *outputs, const Scalar *inputs,
                          const Scalar *gamma, const Scalar *beta,
                          const Scalar *mean, const Scalar *variance,
                          Scalar epsilon) noexcept
{
    for (int c = 0; c < channels; ++c)
    {
        for (int i = 0; i < height * width; ++i)
        {
            int idx = i * channels + c;
            outputs[idx] = gamma[c] * ((inputs[idx] - mean[c]) / std::sqrt(variance[c] + epsilon)) +
                           beta[c];
        }
    }
}

//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// 


template <typename Scalar = double>
auto cnn2_2(const std::array<std::array<std::array<Scalar, 1>, 8>, 8>& initial_input) {

    constexpr int flat_size = 64; 
    std::array<Scalar, flat_size> model_input;
    int idx = 0;
    for (int i0 = 0; i0 < 8; i0++) {
      for (int i1 = 0; i1 < 8; i1++) {
            for (int i2 = 0; i2 < 1; i2++) {
                int flatIndex = i0 * 8 + i1 * 1 + i2 * 1;
                model_input[flatIndex] = initial_input[i0][i1][i2];
            }
        }
    }
    if (model_input.size() != 64) { throw std::invalid_argument("Invalid input size. Expected size: 64"); }


//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// 


    // Layer 1: DepthwiseConv2D
    constexpr std::array<Scalar, 9> depthwiseKernel_1 = {1.001960486e-01, -3.003485203e-01, 4.250909090e-01, -1.947651356e-01, -3.946200609e-01, -2.810938656e-01, 2.207326889e-01, -2.340266407e-01, 5.655122399e-01};
    constexpr std::array<Scalar, 1> depthwiseBias_1 = {-1.593796769e-04};

    // Layer 2: Normalization
    constexpr std::array<Scalar, 1> gamma_2 = {1.005230188e+00};
    constexpr std::array<Scalar, 1> beta_2 = {-1.219598297e-02};
    constexpr std::array<Scalar, 1> mean_2 = {-1.857671514e-02};
    constexpr std::array<Scalar, 1> variance_2 = {8.719730377e-01};
    constexpr Scalar epsilon_2 = 1.000000000e-03;

    // Layer 4: Conv2D
    constexpr std::array<Scalar, 8> convKernel_4 = {-1.063982025e-01, -1.118806303e-01, 2.975300141e-02, -3.881293908e-02, -5.562047362e-01, 7.938609123e-01, -4.131311178e-01, 8.692816645e-02};
    constexpr std::array<Scalar, 8> convBias_4 = {-5.281267222e-04, -1.257273834e-04, 4.418847384e-04, -1.911214902e-03, 4.262214134e-05, -1.893102453e-04, 1.326150377e-04, 8.696930017e-04};

    // Layer 5: Normalization
    constexpr std::array<Scalar, 8> gamma_5 = {9.965236783e-01, 9.979690909e-01, 1.009322643e+00, 9.990699887e-01, 9.957818985e-01, 1.010322452e+00, 9.898195267e-01, 1.003445983e+00};
    constexpr std::array<Scalar, 8> beta_5 = {-3.410758218e-03, 4.008693621e-03, -8.794159628e-03, 1.656205714e-04, -8.793595480e-04, -9.977445006e-03, -1.098197978e-02, -4.273693543e-03};
    constexpr std::array<Scalar, 8> mean_5 = {-6.200084928e-03, -6.371109281e-03, 1.506646164e-03, -2.259885659e-03, -3.141620383e-02, 4.432131350e-02, -2.347024716e-02, 4.808879457e-03};
    constexpr std::array<Scalar, 8> variance_5 = {8.606605530e-01, 8.606994152e-01, 8.600920439e-01, 8.601291776e-01, 8.756881952e-01, 8.911934495e-01, 8.687844872e-01, 8.604140878e-01};
    constexpr Scalar epsilon_5 = 1.000000000e-03;

    // Layer 7: SeparableConv2D
    constexpr std::array<Scalar, 72> sepDepthwise_7 = {-2.275203615e-01, -1.198875457e-01, 1.818154454e-01, -2.546190321e-01, 1.608686298e-01, 4.017106816e-02, 1.419651508e-01, -1.275664419e-01, -1.317406893e-01, 9.581951424e-03, -1.289526969e-01, 6.239905488e-03, 1.109840125e-01, 9.100117534e-02, 9.914262593e-02, -2.466542572e-01, -2.268428206e-01, -4.333589599e-02, -1.286715716e-01, -1.363409758e-01, 2.293604016e-01, -2.233772911e-02, -7.932709157e-02, -3.287924081e-02, 2.271129489e-01, 2.652567923e-01, -2.444286197e-01, -3.506147116e-02, 1.822413504e-01, -1.565704197e-01, -2.034859918e-02, 1.607200503e-01, 1.768548787e-01, -1.393842138e-02, 1.832517385e-01, 1.675387919e-01, -5.780830979e-02, -2.019776404e-01, -1.604269445e-01, -9.809482843e-02, -2.029107213e-01, -5.371919274e-02, 2.510019839e-01, -5.100756139e-02, -1.529874504e-01, -1.366972029e-01, -7.861530781e-02, 7.072952390e-02, 2.413760722e-01, 1.614435762e-01, -9.905675054e-02, -1.777178235e-02, 1.461153328e-01, 1.959684193e-01, 2.646727860e-01, -1.393623203e-01, -4.486200586e-02, 1.090059131e-01, -1.568492353e-01, -2.370953709e-01, -6.905268878e-03, -2.072266787e-01, -2.022864223e-01, 2.407629341e-01, 1.686357558e-01, -7.747425884e-02, -2.610965669e-01, -9.531940520e-02, 4.114380572e-03, -1.667304039e-01, -1.341511160e-01, 2.718350291e-02};
    constexpr std::array<Scalar, 128> sepPointwise_7 = {-1.894312799e-01, 4.752152264e-01, -3.572727144e-01, -1.800029427e-01, -2.716872990e-01, 1.030680165e-01, 1.588171571e-01, -6.560494658e-03, 2.953341901e-01, -2.584357262e-01, -1.817301661e-01, 2.152516991e-01, 1.031372417e-02, 2.624248862e-01, 2.175570577e-01, 2.262110710e-01, -4.051142037e-01, -3.440107405e-01, 3.006349802e-01, 4.313946962e-01, 1.674416810e-01, -3.173368275e-01, 1.303217411e-01, 1.871752739e-01, -4.027310610e-01, -6.972111762e-02, -3.942038119e-01, -4.135297239e-01, -4.838501811e-01, -2.136365175e-01, 2.995010912e-01, -1.530716568e-01, 1.055516526e-01, -2.790776491e-01, 1.145201996e-01, 3.401929438e-01, 2.300055176e-01, -1.685296595e-01, 1.384174228e-01, 4.398273677e-02, 7.799994946e-02, -1.223231573e-02, -3.224312663e-01, 1.874169856e-01, -1.251744181e-01, 4.810268581e-01, 3.352458775e-01, -7.349153608e-02, 4.198209345e-01, -4.147021100e-02, 4.760994315e-01, 2.588134110e-01, 1.032514945e-01, 4.648819566e-01, -1.954188384e-02, -2.875905037e-01, 4.542958438e-01, 1.599450111e-01, 1.589630395e-01, -1.691547967e-02, -4.340677261e-01, 1.184620261e-01, 1.095520481e-01, 3.095618784e-01, 9.078057110e-02, -1.960815676e-02, 2.185022682e-01, 3.052325174e-02, 1.460666060e-01, -2.909825556e-02, -3.111843765e-01, 3.510984033e-02, 3.490378335e-02, 2.000398785e-01, -2.820551991e-01, 3.944379389e-01, 1.130440980e-01, 1.928026788e-02, 1.438929737e-01, 4.726531506e-01, 1.475695819e-01, 1.618747711e-01, -1.479607970e-01, 5.124092847e-02, -5.258032307e-02, -3.171335161e-02, -1.438387036e-01, 1.545767784e-01, -4.052985311e-01, 2.144664973e-01, 1.538257003e-01, -4.445633590e-01, -3.999684379e-02, 3.273868859e-01, -4.433100820e-01, -4.263812676e-02, -2.877120301e-02, -3.944071531e-01, 2.432674766e-01, -6.611194462e-02, -4.800969660e-01, 2.571729422e-01, -2.986300290e-01, -1.430950016e-01, 4.972768128e-01, 7.063967735e-02, 3.018547595e-01, 1.224305406e-01, 4.648581147e-01, -3.334824145e-01, -2.903257608e-01, -2.562323511e-01, 4.110494852e-01, 2.167629004e-01, 2.087504119e-01, -1.045123413e-01, 1.639218479e-01, -2.918227017e-01, -5.590993911e-03, -4.923872054e-01, 1.374061108e-01, 4.508081675e-01, 3.820251524e-01, 3.146770410e-03, 4.361605048e-01, -4.184778631e-01, 1.157567576e-01, 2.024511397e-01};
    constexpr std::array<Scalar, 16> sepPointwiseBias_7 = {7.958608330e-05, 1.817516604e-04, -4.908030387e-04, 5.177714047e-05, 1.993504266e-04, 6.979101454e-05, -5.863014958e-04, 9.815712838e-05, -2.608829891e-05, -2.221458089e-05, 1.743324174e-05, -1.061463699e-04, 5.723306822e-05, 7.634663234e-06, 4.883388829e-05, 2.815359403e-05};

    // Layer 8: Normalization
    constexpr std::array<Scalar, 16> gamma_8 = {9.945276380e-01, 9.915803671e-01, 9.940333366e-01, 9.916538000e-01, 9.867311716e-01, 9.994745255e-01, 9.902711511e-01, 9.951175451e-01, 9.901550412e-01, 9.956187606e-01, 1.001106977e+00, 1.004588485e+00, 9.985836744e-01, 9.981996417e-01, 1.007558942e+00, 1.004770637e+00};
    constexpr std::array<Scalar, 16> beta_8 = {5.318183685e-04, -6.944869645e-03, -2.407350577e-03, -1.013796683e-02, -2.850946039e-03, -4.716398194e-03, -1.225391403e-02, -1.153219491e-02, -1.308120415e-02, -2.426740713e-03, -6.370763294e-03, 4.974403419e-05, -4.763176548e-05, -3.814658849e-03, 5.394484848e-03, -8.107136004e-03};
    constexpr std::array<Scalar, 16> mean_8 = {-1.661159471e-02, -2.068520291e-03, -7.736480329e-04, -1.273463364e-03, 7.593157236e-03, -9.176163934e-03, 1.526152482e-03, 7.994737476e-03, -6.661457475e-03, -9.419500828e-03, -2.170364000e-02, 1.402476244e-02, -3.721783869e-03, -9.174234234e-03, 1.760999300e-02, 8.012004197e-03};
    constexpr std::array<Scalar, 16> variance_8 = {8.620173335e-01, 8.621209264e-01, 8.622249365e-01, 8.614841104e-01, 8.609169126e-01, 8.609470129e-01, 8.608353138e-01, 8.651950359e-01, 8.660311103e-01, 8.618222475e-01, 8.634147644e-01, 8.648886085e-01, 8.654028773e-01, 8.696157932e-01, 8.641026616e-01, 8.610709310e-01};
    constexpr Scalar epsilon_8 = 1.000000000e-03;

    // Layer 10: SeparableConv2D
    constexpr std::array<Scalar, 144> sepDepthwise_10 = {1.717299372e-01, 1.469897032e-01, 1.549475640e-01, -4.455070570e-02, 4.697349668e-02, 3.483610973e-02, -1.442248374e-01, 4.198309034e-02, 1.037338004e-01, -1.050037444e-01, 1.492197718e-02, -1.801160425e-01, 7.414782047e-02, 1.078510471e-02, 2.652216330e-02, -7.145349681e-02, 4.857049510e-02, -1.409256756e-01, -1.350927502e-01, -1.555168908e-02, -4.969048314e-03, -5.154987052e-02, -1.039584130e-01, 1.934445351e-01, 1.391083300e-01, -5.265564844e-02, 1.716608554e-01, -6.114682183e-02, 7.179351896e-02, -1.399647444e-01, 1.933552027e-01, -1.105689779e-01, -5.969741195e-02, -1.769255400e-01, 9.965506941e-02, -8.253284544e-02, -1.180642247e-01, -2.692709491e-02, 1.486581713e-01, -3.681431990e-03, 9.801036865e-02, -1.872372776e-01, 1.507705301e-01, -1.580799073e-01, 1.067276150e-01, 1.954877824e-01, -1.054659560e-01, 1.159936264e-01, 1.359741390e-01, -8.894016594e-02, -1.646351069e-01, -6.549681723e-02, 1.068375111e-01, 1.078093126e-01, 5.836116523e-02, 1.065227017e-02, 1.624035388e-01, -9.751975536e-02, -3.754281625e-02, -1.075673550e-01, 3.154568747e-02, -4.255695269e-02, 3.591690585e-02, 1.311430335e-02, -6.290757656e-02, 6.068501389e-04, -1.872662008e-01, 3.298161551e-03, -1.291509718e-01, 1.054547131e-01, -1.033741608e-01, 1.426997483e-01, -9.731908888e-02, 1.230752841e-01, 1.200391948e-01, -3.729125857e-02, -1.958664507e-01, -5.466848984e-02, -6.151651964e-02, 6.883273274e-02, -7.595256716e-02, -6.743835658e-02, -9.234914929e-02, -1.320492774e-01, 1.307515651e-01, 1.259071529e-01, -1.818121523e-01, -1.514349729e-01, 9.483274817e-02, -3.730304912e-02, -1.820953935e-01, -1.038407534e-01, -1.785520278e-02, 7.452799939e-03, -1.111534014e-01, 1.270239125e-03, 9.389291517e-03, -1.592648625e-01, 1.051516458e-02, 3.936662152e-02, -1.146720052e-01, -3.867389634e-02, -5.000998452e-02, 2.908790670e-02, 1.594412923e-01, 8.896993846e-02, 6.112033874e-02, 5.454202741e-02, -1.248997636e-03, 1.463457495e-01, -5.384289660e-03, 5.290432647e-02, 1.102598533e-01, -1.356078126e-02, -6.272723526e-02, 1.196662337e-01, -1.173673347e-01, -2.800820954e-02, -9.403914958e-02, -6.964098662e-02, -4.152712226e-02, 1.379520595e-01, -1.021908969e-01, 7.874372602e-02, -2.850684803e-03, -6.122590601e-02, 9.562376142e-02, -1.584719568e-01, -1.788541526e-01, 1.536344141e-01, -8.025958389e-02, -1.447961777e-01, 1.254645139e-01, 1.384524703e-01, 4.232059047e-02, 5.407519266e-02, -1.526544690e-01, -1.747266203e-01, 8.047356270e-03, -1.448410898e-01, -3.400837013e-04, 5.872183293e-02, 7.841049135e-02, 1.239248812e-01};
    constexpr std::array<Scalar, 256> sepPointwise_10 = {-1.940563023e-01, -1.052334234e-01, -1.993503273e-01, -1.109661311e-01, 6.364140660e-02, -3.243711591e-01, 3.147680461e-01, 1.656932235e-01, -3.353024721e-01, 2.499480247e-01, -4.273796678e-01, 2.738830447e-01, 8.180164546e-02, 1.682243645e-01, -3.763703704e-01, -3.456470966e-01, -2.163491100e-01, -3.522868454e-01, -3.014266491e-02, -2.643170655e-01, -1.399201751e-01, 2.827020288e-01, 2.051669657e-01, 2.263114005e-01, -5.034635961e-02, 1.788410097e-01, 1.423297077e-01, 1.331447214e-01, 1.985436827e-01, -2.219946831e-01, -2.902749479e-01, 3.115534484e-01, -5.054227263e-02, 3.929403424e-01, -2.165881693e-01, 1.672348827e-01, 1.231653765e-01, -3.073754013e-01, 1.184644029e-01, -2.786146402e-01, -1.451411396e-01, 1.316543967e-01, 3.660258949e-01, 3.803288341e-01, -7.763601094e-02, 3.771652281e-01, -3.554860950e-01, 5.943300202e-02, -3.438498080e-01, 3.722849488e-01, -3.878430128e-01, -1.060232297e-01, -5.473495275e-02, -3.995023295e-02, 2.760932446e-01, -4.099057615e-02, -2.972970344e-02, -2.067879289e-01, -1.683816612e-01, -2.854088880e-02, 3.558395207e-01, 3.117693365e-01, -2.252827138e-01, 1.282046288e-01, -1.139317453e-02, -7.059392054e-03, 7.031388581e-02, -5.509686470e-02, -4.037860334e-01, 1.320607364e-01, 3.229720891e-01, -5.708980188e-02, 1.048050914e-02, 3.728502616e-02, 2.343945503e-01, -4.140996039e-01, 9.135736153e-03, 5.054939166e-02, 3.657532632e-01, 1.872058362e-01, -1.275804453e-02, 2.297059298e-01, -3.005690277e-01, -3.331242800e-01, -2.833525091e-02, -2.700520754e-01, 6.507235765e-02, 1.974313855e-01, -1.256016940e-01, 2.253596038e-01, -3.884098232e-01, 1.311178803e-01, -2.695920169e-01, 4.084829390e-01, 4.066579342e-01, 4.245397449e-01, -3.358662426e-01, -4.308936894e-01, 2.422705144e-01, -4.332507253e-01, -2.359629422e-01, 1.160098389e-01, -2.792380750e-01, 4.151936248e-02, -2.867970765e-01, 5.549771711e-02, 2.177938819e-01, 6.942626089e-02, 3.155413568e-01, -7.527522743e-02, 3.892744184e-01, -3.256881237e-01, -4.124265909e-01, 2.554563284e-01, 3.027190864e-01, -2.426882684e-01, 2.787014544e-01, 2.261395752e-01, -1.111738533e-01, -7.622777671e-02, -1.922323257e-01, -8.913901448e-02, -2.047450989e-01, -1.329014301e-01, -1.091403067e-01, -3.157511652e-01, -8.599881828e-02, -3.083261549e-01, -2.929966748e-01, 2.170293778e-01, -4.291108549e-01, -1.601839811e-01, 1.627963781e-01, -2.593348324e-01, -1.921248883e-01, 2.400531918e-01, -2.887520790e-01, -7.126921415e-02, 2.495991811e-02, 2.387894839e-01, -1.346356869e-01, -2.625882439e-02, -1.144958846e-02, 4.126856923e-01, 1.466958374e-01, -1.144520715e-01, 4.123976231e-01, 1.811932623e-01, 1.709720641e-01, -3.852150142e-01, -3.888009861e-02, -5.706910044e-02, -2.326975614e-01, 3.002304137e-01, 1.831906289e-01, -4.043964446e-01, 2.732847445e-02, -5.021975189e-02, -4.016161263e-01, 1.343986839e-01, -2.381161973e-02, 3.687447608e-01, 2.360620648e-01, -7.041433454e-02, -4.611011222e-02, -2.911348939e-01, 4.428230599e-02, 3.104483187e-01, -4.357557595e-01, 2.811920345e-01, 2.951907516e-01, -2.246998996e-01, -1.063477546e-01, 3.586651683e-01, -1.453491114e-02, -1.370665878e-01, -1.562082320e-01, 2.116755955e-02, -1.627650410e-01, -3.622088581e-02, -1.701852679e-01, 3.995737731e-01, -2.791646123e-02, -4.730978608e-02, -2.858419158e-02, -2.597912252e-01, 2.535098195e-01, -9.269805253e-02, -4.076784253e-01, -3.246568441e-01, -1.457265206e-02, -4.070344865e-01, 9.364727139e-02, 9.788069129e-02, -3.902381659e-01, -2.434639186e-01, -2.659330070e-01, 3.423132300e-01, 1.934083849e-01, 1.984370686e-02, -2.314486653e-01, -3.162897825e-01, 2.158321738e-01, -1.664698571e-01, 5.032680556e-02, -4.036802351e-01, -4.057105482e-01, -4.307743609e-01, -1.572933495e-01, -6.372782588e-02, 1.442502290e-01, 4.046891630e-01, -6.188345701e-02, -1.037052870e-01, 4.028966129e-01, -2.636035085e-01, -3.009636998e-01, -3.434192389e-02, 4.249471724e-01, -3.823003173e-02, 2.032586932e-01, -4.037410319e-01, -3.262847662e-01, -3.888077140e-01, 9.045701474e-02, -2.703487575e-01, 3.342077136e-01, 2.866184413e-01, -1.393090654e-02, -3.571369350e-01, -3.233528733e-01, -4.146727920e-01, -2.750200927e-01, -1.710352749e-01, 9.155491740e-02, -7.950001955e-02, 1.049256623e-01, -3.259475231e-01, 3.548185825e-01, 4.315366745e-01, 2.940897644e-01, -1.258333474e-01, 1.822618544e-01, -4.155123234e-01, 3.937832117e-01, -8.537364006e-02, -1.498542130e-01, 8.582864702e-02, -1.644185036e-01, -3.314950168e-01, -2.575216889e-01, 2.678811550e-01, -1.682281643e-01, -1.222058013e-01, 1.602182686e-01, 3.466060162e-01};
    constexpr std::array<Scalar, 16> sepPointwiseBias_10 = {3.006049083e-04, -2.133943199e-04, -4.775749403e-05, -8.644143818e-04, 9.678461356e-05, 1.150898461e-04, -8.074625512e-04, 9.226156399e-04, -2.798839705e-04, 1.607294544e-04, 1.901537908e-04, -1.937239722e-04, 3.324269783e-04, 2.507878526e-04, -2.863837472e-05, 3.876684787e-07};

    // Layer 11: Normalization
    constexpr std::array<Scalar, 16> gamma_11 = {1.004461169e+00, 9.980822206e-01, 9.940327406e-01, 1.000885606e+00, 9.865493178e-01, 9.874051809e-01, 9.900764823e-01, 9.864392281e-01, 9.930619597e-01, 1.005298615e+00, 9.939231873e-01, 9.982138872e-01, 9.861317873e-01, 9.908013344e-01, 9.907390475e-01, 9.878824353e-01};
    constexpr std::array<Scalar, 16> beta_11 = {3.840599209e-03, -2.350578317e-03, -4.211605061e-03, 1.190853072e-03, -1.375910267e-02, -1.250156574e-02, -9.866967797e-03, -1.384269539e-02, -7.034247275e-03, 4.068916664e-03, -6.305439398e-03, -1.519152080e-03, -1.387705468e-02, -9.246581234e-03, -1.017178968e-02, -1.255196519e-02};
    constexpr std::array<Scalar, 16> mean_11 = {6.599050015e-03, 1.280540135e-02, -2.723134123e-03, -2.496471163e-03, 1.622419991e-02, -2.371120639e-02, -8.799904026e-03, 1.435001846e-02, -7.810087875e-03, 3.314073430e-03, -3.164149821e-02, 2.386093838e-03, -8.364176378e-03, 4.794072825e-03, 1.607301272e-02, 2.096220292e-02};
    constexpr std::array<Scalar, 16> variance_11 = {8.632963300e-01, 8.649020791e-01, 8.654305339e-01, 8.636135459e-01, 8.623440862e-01, 8.646979928e-01, 8.622164726e-01, 8.625158668e-01, 8.630459905e-01, 8.621244431e-01, 8.635728359e-01, 8.634645343e-01, 8.619357944e-01, 8.658362031e-01, 8.658704758e-01, 8.683056831e-01};
    constexpr Scalar epsilon_11 = 1.000000000e-03;

    // Layer 13: GlobalAveragePooling2D
    // Dense layer 14
    constexpr std::array<Scalar, 80> weights_14 = {4.073008299e-01, 2.981559932e-01, -4.104058072e-02, 5.829665810e-02, -8.241007477e-02, 2.595603168e-01, 2.929798067e-01, -2.275074422e-01, 2.219552398e-01, -5.032736436e-02, 4.536517262e-01, 4.696661234e-02, -5.013157129e-01, -4.160146415e-01, -3.366550803e-01, -4.967922270e-01, 5.040151477e-01, 3.519393802e-01, -4.119922221e-01, 2.868688107e-01, 1.537361741e-01, 8.358039707e-02, -2.924937941e-02, 3.716203272e-01, 4.219487309e-01, 1.469447166e-01, -4.007726312e-01, -2.513175309e-01, 2.055896223e-01, 2.367607504e-01, 1.896027923e-01, -2.851406932e-01, 3.021302521e-01, -2.287694067e-01, 3.551514745e-01, 4.821354747e-01, 4.815050215e-02, -1.485974044e-01, -1.767424308e-02, 5.128350854e-01, 4.647957534e-02, -1.906232983e-01, 3.685574532e-01, -8.824869245e-02, 4.387775362e-01, 7.590501755e-02, 4.851160944e-01, 1.225465834e-01, 1.554915961e-02, 9.733609855e-02, -1.632785052e-01, 3.418226242e-01, 2.684315145e-01, 4.546738267e-01, 4.525983036e-01, -3.231490254e-01, -2.593229413e-01, 2.903413177e-01, -9.581410885e-02, 1.842256039e-01, -2.149466984e-02, 1.779597998e-01, -1.013906896e-01, 3.696494997e-01, 4.120369256e-01, -3.291851580e-01, -1.010589302e-01, -2.624639571e-01, -4.865613878e-01, 2.309296429e-01, -2.897442318e-02, -2.878875732e-01, 6.979771703e-02, -5.474642292e-02, 3.435767069e-02, 1.055654064e-01, -1.885283440e-01, -3.318193853e-01, 1.278941780e-01, 1.274130791e-01};
    constexpr std::array<Scalar, 5> biases_14 = {-2.393361181e-03, 7.217437960e-03, 1.232613157e-02, 2.909902949e-03, -1.314988639e-02};


//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// 


    auto relu = +[](Scalar& output, Scalar input, Scalar alpha) noexcept {
        output = input > 0 ? input : 0;
    };

    auto linear = +[](Scalar& output, Scalar input, Scalar alpha) noexcept {
        output = input;
    };

    auto softmax = +[](Scalar *output, Scalar *input, int size) noexcept {
        Scalar max_val = *std::max_element(input, input + size);
        Scalar sum = 0;
        for (int i = 0; i < size; ++i)
        {
            const Scalar exp_val = std::exp(input[i] - max_val);
            output[i] = exp_val;
            sum += exp_val;
        }
        for (int i = 0; i < size; ++i)
        {
            output[i] /= sum;
        }
    };

//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// 

    // depthwiseConv2DForward call for layer 1
    std::array<Scalar, (8 * 8 * 1)> layer_1_output;
    depthwiseConv2DForward(
        layer_1_output.data(), model_input.data(),
        depthwiseKernel_1.data(), depthwiseBias_1.data(),
        1, 8, 8,
        1, 8, 8,
        3, 3, 1, 1, 1, 1,
        linear, 0.0);

    std::array<Scalar, (8 * 8 * 1)> layer_2_output;
    batchNormalization2D<Scalar, 1, 8, 8>(
        layer_2_output.data(), layer_1_output.data(),
        gamma_2.data(), beta_2.data(),
        mean_2.data(), variance_2.data(),
        epsilon_2);

    // Pure activation layer 3
    std::array<Scalar, 64> layer_3_output;
    for (int i = 0; i < 64; ++i) {
        relu(layer_3_output[i], layer_2_output[i], 0.0);
    }

    // conv2DForward call for layer 4
    std::array<Scalar, (8 * 8 * 8)> layer_4_output;
    conv2DForward<Scalar, 8, 8, 8>(
        layer_4_output.data(), layer_3_output.data(),
        convKernel_4.data(), convBias_4.data(),
        1, 8, 8,
        1, 1, 1, 1, 0, 0,
        linear, 0.0);

    std::array<Scalar, (8 * 8 * 8)> layer_5_output;
    batchNormalization2D<Scalar, 8, 8, 8>(
        layer_5_output.data(), layer_4_output.data(),
        gamma_5.data(), beta_5.data(),
        mean_5.data(), variance_5.data(),
        epsilon_5);

    // Pure activation layer 6
    std::array<Scalar, 512> layer_6_output;
    for (int i = 0; i < 512; ++i) {
        relu(layer_6_output[i], layer_5_output[i], 0.0);
    }

    // separableConv2DForward call for layer 7
    std::array<Scalar, (8 * 8 * 16)> layer_7_output;
    separableConv2DForward<Scalar, 16, 8, 8>(
        layer_7_output.data(), layer_6_output.data(),
        sepDepthwise_7.data(), sepPointwise_7.data(), sepPointwiseBias_7.data(),
        8, 8, 8,
        3, 3, 1, 1, 1, 1,
        linear, 0.0);

    std::array<Scalar, (8 * 8 * 16)> layer_8_output;
    batchNormalization2D<Scalar, 16, 8, 8>(
        layer_8_output.data(), layer_7_output.data(),
        gamma_8.data(), beta_8.data(),
        mean_8.data(), variance_8.data(),
        epsilon_8);

    // Pure activation layer 9
    std::array<Scalar, 1024> layer_9_output;
    for (int i = 0; i < 1024; ++i) {
        relu(layer_9_output[i], layer_8_output[i], 0.0);
    }

    // separableConv2DForward call for layer 10
    std::array<Scalar, (8 * 8 * 16)> layer_10_output;
    separableConv2DForward<Scalar, 16, 8, 8>(
        layer_10_output.data(), layer_9_output.data(),
        sepDepthwise_10.data(), sepPointwise_10.data(), sepPointwiseBias_10.data(),
        16, 8, 8,
        3, 3, 1, 1, 1, 1,
        linear, 0.0);

    std::array<Scalar, (8 * 8 * 16)> layer_11_output;
    batchNormalization2D<Scalar, 16, 8, 8>(
        layer_11_output.data(), layer_10_output.data(),
        gamma_11.data(), beta_11.data(),
        mean_11.data(), variance_11.data(),
        epsilon_11);

    // Pure activation layer 12
    std::array<Scalar, 1024> layer_12_output;
    for (int i = 0; i < 1024; ++i) {
        relu(layer_12_output[i], layer_11_output[i], 0.0);
    }

    // globalAvgPooling2D call for layer 13
    std::array<Scalar, 16> layer_13_output;
    globalAvgPooling2D(
        layer_13_output.data(), layer_12_output.data(), 8, 8, 16);

    std::array<Scalar, 5> layer_14_output;
    Dense<Scalar, 5>(
        layer_14_output.data(), layer_13_output.data(),
        weights_14.data(), biases_14.data(),
        16, linear, 0.0);

    // Standalone softmax layer for layer 14
    softmax(layer_14_output.data(), layer_14_output.data(), 5);

    // final placeholder
    std::array<Scalar, 5> model_output = layer_14_output;

    return model_output;
}
