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

template <typename Scalar, int size>
void LayerNormalization(Scalar *outputs, const Scalar *inputs, const Scalar *gamma, const Scalar *beta, Scalar epsilon) noexcept
{
    Scalar mean = 0;
    Scalar variance = 0;
    for (int i = 0; i < size; ++i)
    {
        mean += inputs[i];
    }
    mean /= size;
    for (int i = 0; i < size; ++i)
    {
        variance += (inputs[i] - mean) * (inputs[i] - mean);
    }
    variance /= size;
    for (int i = 0; i < size; ++i)
    {
        outputs[i] = gamma[i] * ((inputs[i] - mean) / std::sqrt(variance + epsilon)) + beta[i];
    }
}

template <typename Scalar, int size>
void BatchNormalization(Scalar *outputs, const Scalar *inputs, const Scalar *gamma, const Scalar *beta, const Scalar *mean, const Scalar *variance, const Scalar epsilon) noexcept
{
    for (int i = 0; i < size; ++i)
    {
        outputs[i] = gamma[i] * ((inputs[i] - mean[i]) / std::sqrt(variance[i] + epsilon)) + beta[i];
    }
}

//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// 


template <typename Scalar = double>
auto dense1(const std::array<Scalar, 3>& initial_input) {

    constexpr int flat_size = 3; 
    std::array<Scalar, flat_size> model_input;
    int idx = 0;
        for (int i=0; i<flat_size; i++) { model_input[i] = initial_input[i]; }
    if (model_input.size() != 3) { throw std::invalid_argument("Invalid input size. Expected size: 3"); }


//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// 


    // Dense layer 3
    constexpr std::array<Scalar, 24> weights_3 = {-1.190937404e-02, -5.307764411e-01, 7.198203206e-01, 5.320910215e-01, -4.846768081e-01, -4.450701773e-01, -2.095842361e-01, 6.298508644e-01, -4.987424016e-01, -6.414932013e-01, 2.039552182e-01, -3.761289120e-01, 6.682286859e-01, -7.538147569e-01, 6.114871502e-01, -2.381901592e-01, -3.102867901e-01, -3.076702952e-01, 2.247886732e-02, 6.096379757e-01, -1.710423678e-01, 3.925664425e-01, 6.397294998e-01, 4.580197632e-01};
    constexpr std::array<Scalar, 8> biases_3 = {-3.120125458e-02, 4.158504680e-02, -6.706858426e-02, -7.746756077e-02, 4.355937243e-02, 2.755216137e-02, 4.671900347e-02, 1.411576271e-01};

    // Dense layer 4
    constexpr std::array<Scalar, 64> weights_4 = {4.462879598e-01, 2.085821629e-01, 5.953876376e-01, -9.811563045e-02, -8.586118929e-03, 5.415587425e-01, -4.657579064e-01, 1.525907665e-01, 2.961412966e-01, 2.053385675e-01, -6.190593243e-01, 2.621248960e-01, -1.713193953e-01, -4.693469405e-01, 5.267087221e-01, -1.766177714e-01, 5.537845492e-01, -8.060796559e-02, 8.760244399e-02, -2.916286886e-01, 4.876872301e-01, -7.521528006e-02, 5.727753639e-01, 8.984173834e-02, -3.177414834e-02, 5.142145604e-02, 3.313485086e-01, 1.867810488e-01, 5.040286779e-01, 5.500572324e-01, 1.870516986e-01, 3.561371863e-01, 5.315422416e-01, 3.907974064e-01, -4.373283982e-01, 5.579469800e-01, -2.852792144e-01, -1.777767576e-02, 6.218777299e-01, 1.555801034e-01, -3.727714717e-01, -1.786966622e-01, 9.040182829e-02, 3.914789557e-01, 3.843234777e-01, 3.727408946e-01, -5.689059496e-01, 1.456284076e-01, 2.777903676e-01, 2.071400583e-01, -5.090424418e-01, 4.372810125e-01, 1.776434332e-01, -5.857012272e-01, 5.717847347e-01, 5.684506521e-02, -1.819307357e-01, 5.645943880e-01, 3.066464365e-01, 2.175406665e-01, -5.047312379e-01, -3.152227998e-01, -1.415913850e-01, -6.700379774e-03};
    constexpr std::array<Scalar, 8> biases_4 = {1.355462708e-02, 7.926310599e-02, -2.028815448e-02, 6.969305128e-02, -7.053395966e-04, -2.812560648e-02, -9.370187670e-02, -5.444031581e-02};

    // Layer 5: Normalization
    constexpr std::array<Scalar, 8> gamma_5 = {1.009482265e+00, 1.028126717e+00, 1.066814184e+00, 9.641088247e-01, 9.885975718e-01, 1.072773099e+00, 9.531732798e-01, 9.150727987e-01};
    constexpr std::array<Scalar, 8> beta_5 = {-2.006273530e-02, 8.675864339e-02, -8.538980037e-02, 7.681505382e-02, -9.462012351e-02, 8.688245714e-02, -1.058105379e-01, -4.910670593e-02};
    constexpr Scalar epsilon_5 = 1.000000000e-03;

    // Dense layer 6
    constexpr std::array<Scalar, 64> weights_6 = {-4.434689581e-01, -4.147334993e-01, -2.571999133e-01, -1.468741056e-02, -5.556757450e-01, -5.277353525e-01, -4.840237796e-01, -6.437171996e-02, -4.500527680e-02, 4.200557470e-01, -4.032296240e-01, 6.068063974e-01, 5.356696248e-01, 2.168546803e-02, 1.969794929e-01, -2.465195358e-01, -6.371591687e-01, 1.429573148e-01, -4.193763137e-01, -3.898166418e-01, 2.311607264e-02, -3.502175212e-01, 4.555709660e-01, 2.075046748e-01, -4.328397512e-01, -2.494437993e-01, -5.190909505e-01, 7.308096886e-01, -1.878615171e-01, -3.921609819e-01, -4.413656592e-01, -5.457357168e-01, -4.515094757e-01, -8.515742421e-02, 1.834127307e-01, -7.188475132e-02, 1.352307796e-01, -6.052878499e-01, 6.266108751e-01, -2.061433494e-01, -2.473618984e-01, -7.768895477e-02, 4.300865531e-02, 5.698637962e-01, 3.773369789e-01, 3.603057563e-01, -1.038210616e-01, -2.236919552e-01, -4.172343016e-01, -2.832561433e-01, 1.277438086e-02, -1.423892528e-01, 1.474211663e-01, -9.812713414e-02, -2.646853030e-01, -5.536430329e-02, 3.267974555e-01, -5.749552846e-01, 5.257062912e-01, -2.305033356e-01, -3.474723548e-02, 2.856920063e-01, 2.280490994e-01, 2.584920824e-01};
    constexpr std::array<Scalar, 8> biases_6 = {3.536058590e-02, 5.074284226e-02, -2.074950784e-01, 1.171063110e-01, -8.413945884e-02, 1.236582547e-01, -4.365410190e-03, -1.806913130e-02};

    // Layer 7: Normalization
    constexpr std::array<Scalar, 8> gamma_7 = {9.021655917e-01, 1.045012712e+00, 1.437439442e+00, 1.045335531e+00, 9.928500652e-01, 1.232170701e+00, 1.011414409e+00, 9.639930129e-01};
    constexpr std::array<Scalar, 8> beta_7 = {-1.917675138e-01, 1.655071974e-01, 3.373545706e-01, 1.728691608e-01, -2.515123188e-01, 1.108119935e-01, 6.961321086e-02, -3.643534780e-01};
    constexpr std::array<Scalar, 8> mean_7 = {1.829269715e-02, 2.950363755e-01, -4.720874131e-01, -1.847146004e-01, 1.140029579e-01, -2.129251808e-01, -5.035857670e-03, 4.984407499e-02};
    constexpr std::array<Scalar, 8> variance_7 = {7.027350366e-02, 2.152027935e-01, 4.623310268e-02, 4.027276337e-01, 1.289719641e-01, 1.041208021e-02, 5.830404758e-01, 2.758731544e-01};
    constexpr Scalar epsilon_7 = 1.000000000e-03;

    // Dense layer 8
    constexpr std::array<Scalar, 64> weights_8 = {-4.025300145e-01, 1.410606503e-01, -2.362947315e-01, 5.316548944e-01, -1.982306093e-01, 6.635280848e-01, -3.540197313e-01, -1.365163624e-01, 1.941394955e-01, 5.739911795e-01, -4.123908654e-02, -2.176883370e-01, 1.744694710e-01, 2.083249390e-02, 4.102400541e-01, -5.386065319e-02, -5.282450914e-01, -3.580724597e-01, -5.575715899e-01, -1.954497844e-01, -2.588289082e-01, -2.614928484e-01, -1.588486880e-01, -2.784588039e-01, 4.746670127e-01, 4.863028526e-01, 4.954077899e-01, 1.556253880e-01, -2.872191966e-01, -1.870768368e-01, -7.083227634e-01, 6.328939199e-01, -3.824826181e-01, -3.409745991e-01, 3.240706921e-01, 6.213482022e-01, 4.714741558e-02, 1.118114144e-01, -9.954532981e-02, -1.678820252e-01, -1.341346093e-02, 5.326037407e-01, -1.052661911e-01, -5.500003099e-01, 4.519976974e-01, -2.274839133e-01, -1.771484017e-01, 3.049014211e-01, 1.736919284e-01, -1.322346181e-01, 3.232865036e-01, -5.233362317e-01, -1.543697249e-02, 5.021371841e-01, 1.525325030e-01, -4.917078614e-01, -3.023039401e-01, 2.900039256e-01, 1.940631717e-01, 1.726759672e-01, 2.549512088e-01, 2.881493568e-01, -2.425485849e-02, -3.462527990e-01};
    constexpr std::array<Scalar, 8> biases_8 = {-4.169865325e-02, 2.403358221e-01, 7.465521246e-02, -2.135412544e-01, -3.495707214e-01, -2.441853285e-01, 1.869042963e-01, -6.793269888e-03};

    // Dense layer 12
    constexpr std::array<Scalar, 64> weights_12 = {-5.505729318e-01, -1.911525279e-01, -3.935022354e-01, 1.562434435e-01, -5.610117316e-01, 6.975556165e-02, -4.094119743e-02, 2.638524771e-01, -4.041375220e-01, -3.283296525e-01, -3.721309900e-01, 1.667396277e-01, -6.659437418e-01, 1.645655185e-01, 1.350420266e-01, 3.537037075e-01, 4.955300689e-02, 6.109561771e-02, -4.378079250e-02, 2.480117530e-01, -2.778511047e-01, 2.184876949e-01, 2.276191413e-01, 3.068682551e-01, -5.007047951e-02, -5.852149427e-02, -1.236273572e-01, 4.331286252e-02, 1.130069494e-01, 5.748708546e-02, 1.159013901e-02, 1.607727073e-02, 1.642617285e-01, -2.287833393e-01, 2.989144437e-02, -5.533155426e-02, -1.160527319e-01, 4.367905110e-02, 2.299744189e-01, 5.506089330e-02, 4.486713931e-02, -1.865193695e-01, -3.333890438e-01, 3.389824331e-01, -6.282999516e-01, 3.687327206e-01, -1.792488433e-02, 2.226426452e-01, 5.474852920e-01, 1.324458569e-01, 1.617846042e-01, 4.274240434e-01, -6.437104344e-01, 5.438204408e-01, 3.598532081e-01, 5.696235299e-01, -7.727150321e-01, 8.503334597e-03, -3.085468113e-01, -7.461377233e-02, -3.152667582e-01, -2.402154952e-01, -1.486628652e-01, 1.383310705e-01};
    constexpr std::array<Scalar, 8> biases_12 = {-1.085677296e-01, -1.289459467e-01, -3.228612989e-02, 1.005827934e-01, -2.437033355e-01, 1.577459276e-01, 1.045476273e-01, 2.738159299e-01};

    // Layer 13: Normalization
    constexpr std::array<Scalar, 8> gamma_13 = {8.165856600e-01, 7.502728701e-01, 8.793422580e-01, 7.982499003e-01, 8.100761771e-01, 8.276692033e-01, 5.309399962e-01, 8.855851293e-01};
    constexpr std::array<Scalar, 8> beta_13 = {3.827512637e-02, -8.174670488e-02, 1.051426902e-01, -9.452357888e-02, 1.249620691e-01, 2.054515108e-02, 1.249404624e-01, -9.420199320e-03};
    constexpr Scalar epsilon_13 = 1.000000000e-03;

    // Dense layer 14
    constexpr std::array<Scalar, 80> weights_14 = {-2.930999696e-01, -1.799608022e-01, -2.930907011e-01, -5.666048452e-02, 1.370360106e-01, -5.268239975e-01, 3.538134992e-01, -5.774774030e-02, -8.331868798e-02, 4.951301217e-02, 2.247682400e-02, -3.165739775e-01, -2.893388830e-02, -1.455264539e-01, -2.129014768e-02, -1.455134600e-01, 1.456314176e-01, -2.243485600e-01, 3.155744374e-01, -1.175462529e-01, -4.250094295e-02, -2.290289849e-01, -2.444955148e-02, 4.033285677e-01, 2.754957080e-01, 1.361281276e-01, -3.031132221e-01, 3.157068789e-01, -5.053374171e-01, -2.187751532e-01, -2.927198112e-01, 1.537812948e-01, 1.198448613e-01, 5.959691405e-01, -2.255936414e-01, -9.454752319e-03, 3.420618474e-01, 3.656795621e-01, -1.570723802e-01, 5.532554537e-02, -3.162537515e-01, -3.112625778e-01, -2.898939848e-01, -2.975891829e-01, 1.792531461e-01, -4.276719093e-01, 1.347149312e-01, -8.771914057e-03, -1.332701445e-01, -1.051888615e-01, 4.045766890e-01, -2.893493474e-01, 7.668300718e-02, -2.839585245e-01, 2.291847616e-01, 1.676967889e-01, -3.167735934e-01, 3.377088532e-02, 1.356033236e-01, -1.890696883e-01, 2.236148417e-01, 1.644471139e-01, 2.210807353e-01, 4.509641826e-01, 3.540599644e-01, 4.701783657e-01, 4.592218995e-01, -2.525242567e-01, 7.059664726e-01, 1.572286338e-02, -3.787680864e-01, -2.159769982e-01, -1.507566869e-01, -9.949543327e-02, 4.911003411e-01, -4.837934971e-01, 3.417843580e-01, 1.674805880e-01, -2.751919925e-01, 3.464435935e-01};
    constexpr std::array<Scalar, 10> biases_14 = {1.484513283e-01, 9.448259324e-02, -1.153637245e-01, 4.445960745e-02, 1.506462544e-01, 8.086975664e-03, 1.757473797e-01, 1.844954640e-01, 1.352260262e-01, -9.987663478e-02};


//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// 


    auto linear = +[](Scalar& output, Scalar input, Scalar alpha) noexcept {
        output = input;
    };

    auto sigmoid = +[](Scalar& output, Scalar input, Scalar alpha) noexcept {
        output = 1 / (1 + std::exp(-input));
    };

    auto silu = +[](Scalar& output, Scalar input, Scalar alpha) noexcept {
        auto sigmoid = 1 / (1 + std::exp(-input));
        output = input * sigmoid;
    };

    auto tanhCustom = +[](Scalar& output, Scalar input, Scalar alpha) noexcept {
        output = std::tanh(input);
    };

    auto relu = +[](Scalar& output, Scalar input, Scalar alpha) noexcept {
        output = input > 0 ? input : 0;
    };

    auto elu = +[](Scalar& output, Scalar input, Scalar alpha) noexcept {
        output = input > 0 ? input : alpha * (std::exp(input) - 1);
    };

//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// 

    // Pure activation layer 1
    std::array<Scalar, 3> layer_1_output;
    for (int i = 0; i < 3; ++i) {
        relu(layer_1_output[i], model_input[i], 0.0);
    }

    std::array<Scalar, 8> layer_3_output;
    Dense<Scalar, 8>(
        layer_3_output.data(), layer_1_output.data(),
        weights_3.data(), biases_3.data(),
        3, linear, 0.0);

    std::array<Scalar, 8> layer_4_output;
    Dense<Scalar, 8>(
        layer_4_output.data(), layer_3_output.data(),
        weights_4.data(), biases_4.data(),
        8, silu, 0.0);

    std::array<Scalar, 8> layer_5_output;
    LayerNormalization<Scalar, 8>(
        layer_5_output.data(), layer_4_output.data(),
        gamma_5.data(), beta_5.data(),
        epsilon_5);

    std::array<Scalar, 8> layer_6_output;
    Dense<Scalar, 8>(
        layer_6_output.data(), layer_5_output.data(),
        weights_6.data(), biases_6.data(),
        8, tanhCustom, 0.0);

    std::array<Scalar, 8> layer_7_output;
    BatchNormalization<Scalar, 8>(
        layer_7_output.data(), layer_6_output.data(),
        gamma_7.data(), beta_7.data(),
        mean_7.data(), variance_7.data(),
        epsilon_7);

    std::array<Scalar, 8> layer_8_output;
    Dense<Scalar, 8>(
        layer_8_output.data(), layer_7_output.data(),
        weights_8.data(), biases_8.data(),
        8, linear, 0.0);

    // Pure activation layer 9
    std::array<Scalar, 8> layer_9_output;
    for (int i = 0; i < 8; ++i) {
        sigmoid(layer_9_output[i], layer_8_output[i], 0.0);
    }

    // Pure activation layer 11
    std::array<Scalar, 8> layer_11_output;
    for (int i = 0; i < 8; ++i) {
        elu(layer_11_output[i], layer_9_output[i], 1.0);
    }

    std::array<Scalar, 8> layer_12_output;
    Dense<Scalar, 8>(
        layer_12_output.data(), layer_11_output.data(),
        weights_12.data(), biases_12.data(),
        8, linear, 0.0);

    std::array<Scalar, 8> layer_13_output;
    LayerNormalization<Scalar, 8>(
        layer_13_output.data(), layer_12_output.data(),
        gamma_13.data(), beta_13.data(),
        epsilon_13);

    std::array<Scalar, 10> layer_14_output;
    Dense<Scalar, 10>(
        layer_14_output.data(), layer_13_output.data(),
        weights_14.data(), biases_14.data(),
        8, linear, 0.0);

    // final placeholder
    std::array<Scalar, 10> model_output = layer_14_output;

    return model_output;
}
