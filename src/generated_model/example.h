#pragma once
#include <iostream>
#include <array>
#include <random>
#include <cmath>
#include <functional>

template<typename Scalar>
using activationFunction = void(*)(Scalar&, Scalar, Scalar);

//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//

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

template <typename Scalar = double>
auto example(const std::array<Scalar, 3>& initial_input) { 

    constexpr std::array<Scalar, 3> input_norms = {9.859801248e-01, 9.792372050e-01, 9.852146633e-01};

    constexpr std::array<Scalar, 3> input_mins = {3.083498694e-03, 1.103722129e-02, 6.335799082e-03};

    std::array<Scalar, 3> model_input;

    for (int i = 0; i < 3; i++) { model_input[i] = (initial_input[i] - input_mins[i]) / (input_norms[i]); }

    if (model_input.size() != 3) { throw std::invalid_argument("Invalid input size. Expected size: 3"); } 

    //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// 

    constexpr std::array<Scalar, 24> weights_3 = {-6.728608012e-01, -1.313065886e-01, -4.564309418e-01, 4.920005798e-01, -2.154353112e-01, 1.516289860e-01, 1.838508248e-01, 7.470854521e-01, -2.218966484e-01, 3.959499598e-01, 5.321294665e-01, 7.252561860e-03, -2.544699013e-01, 3.781022727e-01, 1.682603061e-01, 2.208171636e-01, 6.066064835e-01, -5.916588306e-01, 5.424890518e-01, -7.106128335e-02, 7.361915708e-01, 2.954361141e-01, -6.237546206e-01, 4.846137762e-01};

    constexpr std::array<Scalar, 8> biases_3 = {1.894135214e-02, 3.202559054e-02, 2.899060212e-02, 1.790066250e-02, -3.134807805e-03, 4.268451408e-02, 1.134215388e-02, -8.661418222e-03};

    constexpr std::array<Scalar, 64> weights_4 = {-2.655526996e-01, 1.016645357e-01, -5.373798013e-01, -4.261870682e-01, 3.206447065e-01, -2.297535688e-01, -1.589301229e-01, 3.723084331e-01, 2.916692495e-01, 4.291096702e-02, -2.918314934e-01, -5.948671699e-01, -3.372852206e-01, 4.478961229e-01, 2.397555560e-01, -2.993493676e-01, 3.915114403e-01, -4.088997245e-01, 3.265275434e-02, -4.433298707e-01, 2.775190175e-01, 6.262307763e-01, 2.933279276e-01, 3.672431409e-01, -2.980503142e-01, -6.377719045e-01, -2.880002260e-01, -3.111260235e-01, 5.145444870e-01, 3.833316267e-01, 3.163295984e-01, 2.556556463e-01, -1.421552300e-01, 1.102467999e-01, 5.777072906e-01, 5.162986517e-01, -3.091693521e-01, -3.597130179e-01, 4.979804754e-01, 3.980865479e-01, -5.185080171e-01, -3.327046633e-01, -4.911856726e-02, -6.257461905e-01, -2.654695511e-01, 2.763748765e-01, -4.182887077e-01, 5.203384161e-01, 2.977734208e-01, 3.110255674e-02, 2.004626542e-01, -3.933978975e-01, 1.644137688e-02, 9.767743200e-02, -4.401362240e-01, -3.191339374e-01, 4.026140273e-01, 3.198238909e-01, -3.551059365e-01, -3.470821083e-01, 3.733520210e-01, 3.574077189e-01, -3.115932047e-01, -4.340291619e-01};

    constexpr std::array<Scalar, 8> biases_4 = {-2.455696277e-02, 1.057051588e-02, 4.669509828e-02, -5.865584686e-02, -1.129978616e-02, 2.808775194e-02, 2.851331141e-03, 1.769026183e-02};

    constexpr std::array<Scalar, 8> gamma_5 = {1.015615940e+00, 1.025802970e+00, 1.033450484e+00, 9.828541875e-01, 9.999902844e-01, 9.924752116e-01, 9.523199797e-01, 1.030056357e+00};

    constexpr std::array<Scalar, 8> beta_5 = {1.198863052e-02, -8.913222700e-02, -6.713096052e-02, 1.317612268e-02, -8.211208507e-03, 3.345802054e-02, 1.089955401e-02, -2.884104848e-03};

    constexpr Scalar epsilon_5 = 1.000000000e-03;

    constexpr std::array<Scalar, 64> weights_6 = {2.008610070e-01, 3.824090958e-01, 3.711136878e-01, 5.961924195e-01, -4.152491093e-01, -3.090492785e-01, -4.871101081e-01, 2.799166441e-01, 3.734789491e-01, 4.925452545e-02, 1.371673942e-01, 4.369363487e-01, 2.707842886e-01, -6.127969623e-01, 2.798981071e-01, -6.059750915e-01, 9.031183273e-02, 4.850766063e-01, 9.583407082e-04, -5.182337388e-02, 3.826343417e-01, -4.503844082e-01, -4.887508973e-02, -6.338075548e-02, 3.802551925e-01, -3.452078700e-01, -2.837289274e-01, -7.527941465e-02, 5.774675012e-01, 3.711174726e-01, -4.912964702e-01, 1.668628007e-01, -1.848005950e-01, 5.883837342e-01, -1.217612624e-01, 4.280028939e-01, 2.979469001e-01, -2.477812618e-01, -3.541867137e-01, 2.106873989e-01, -8.135163784e-02, 2.832155526e-01, -5.984395742e-01, -2.737661302e-01, 2.388887405e-01, 4.277097881e-01, 3.972464502e-01, 3.567474186e-01, 1.008749455e-01, -2.764050364e-01, 2.506539822e-01, -6.016538665e-02, -4.940406382e-01, -4.033833146e-01, 2.158156782e-01, 4.379447103e-01, -2.425323725e-01, -5.139911771e-01, -3.017818630e-01, -3.575763106e-01, -2.520323098e-01, 6.312535703e-02, 2.100877613e-01, -5.277147889e-01};

    constexpr std::array<Scalar, 8> biases_6 = {-2.850451879e-02, -2.489393018e-02, -2.337823808e-02, 1.896272600e-02, -2.753569558e-02, 7.738285512e-02, -1.123202685e-02, 6.336874515e-02};

    constexpr std::array<Scalar, 8> gamma_7 = {1.009676456e+00, 1.036532521e+00, 9.009156823e-01, 1.012381196e+00, 1.004805326e+00, 9.721708298e-01, 9.429966807e-01, 1.012142777e+00};

    constexpr std::array<Scalar, 8> beta_7 = {3.218593001e-01, 2.474364191e-01, 2.716482282e-01, -1.943210959e-01, 1.839101017e-01, 4.712961987e-02, 9.850335866e-02, 9.146582335e-02};

    constexpr std::array<Scalar, 8> mean_7 = {-7.696190476e-01, 3.570911288e-01, -8.251363635e-01, -4.614270329e-01, -1.997334212e-01, 8.094437122e-01, 6.294999123e-01, 3.431711197e-01};

    constexpr std::array<Scalar, 8> variance_7 = {2.114844136e-02, 4.081488848e-01, 5.852068309e-03, 1.359766424e-01, 1.961908489e-01, 1.306319889e-02, 3.669273481e-02, 3.319759071e-01};

    constexpr Scalar epsilon_7 = 1.000000000e-03;

    constexpr std::array<Scalar, 64> weights_8 = {6.343092322e-01, -5.072917938e-01, -3.001285791e-01, -1.943929493e-01, 5.468881726e-01, -4.600484669e-01, -2.839317620e-01, -7.313191891e-02, -1.682214141e-01, -2.629324794e-01, 8.544393629e-02, -5.419343710e-01, -1.025550961e-01, -2.707498670e-01, -4.116820544e-02, -2.105078697e-01, 3.958990574e-01, -2.294497192e-01, -3.773896694e-01, -4.838137925e-01, 4.364753366e-01, 3.192947209e-01, -2.864552140e-01, 3.655180633e-01, 1.519656479e-01, 7.074348330e-01, -5.994462967e-01, 5.284430981e-01, -3.194083869e-01, 3.568994254e-02, -2.753601670e-01, -1.794094741e-01, 4.187447131e-01, -1.760551184e-01, 4.961210489e-01, -6.713150144e-01, -1.325331628e-01, -1.005992368e-01, 3.337558806e-01, 1.213353276e-01, 1.903844178e-01, -3.628745079e-01, -6.576982141e-01, 2.514625192e-01, 4.005022943e-01, -3.031307459e-01, -3.082392812e-01, -2.268908471e-01, 1.844345480e-01, -3.556323051e-02, 4.848551452e-01, -4.144530892e-01, -8.256339282e-02, 4.231309891e-01, 4.473712146e-01, -1.255500019e-01, -1.190855727e-01, -7.082174532e-03, 2.351879627e-01, -3.734745085e-01, -4.034081995e-01, 3.869620562e-01, 1.939641833e-01, -5.179841518e-01};

    constexpr std::array<Scalar, 8> biases_8 = {-1.282525510e-01, -3.178204596e-01, 1.602094173e-01, -1.888014227e-01, 1.494231373e-01, -1.688864082e-01, 1.978861727e-02, -7.142577320e-02};

    constexpr std::array<Scalar, 64> weights_12 = {-1.769883782e-01, -1.710211160e-03, -4.867954925e-02, 1.044765115e-02, 1.472661346e-01, 1.008204278e-02, -4.835811257e-02, -8.756382018e-02, 2.237122804e-01, -1.654239893e-01, -4.332410693e-01, 2.729918957e-01, -2.400297821e-01, -2.199282311e-02, -2.746744752e-01, -6.916473061e-02, -5.447099209e-01, 6.335625052e-02, -2.340954244e-01, 4.264142513e-01, 4.890418798e-02, -6.793625355e-01, -1.440081000e-01, -4.120460451e-01, 2.254276872e-01, 1.627229452e-01, -4.166041911e-01, -4.500589147e-02, -2.511164248e-01, -1.844502687e-01, -5.052298307e-01, 2.642994225e-01, -5.127264261e-01, 4.662086666e-01, -7.320999354e-02, 6.649609804e-01, 5.357679129e-01, -4.113850296e-01, -7.418540306e-03, -6.039476022e-02, 2.090597451e-01, 3.707290888e-01, 2.335193008e-01, -6.725528091e-02, -8.734042943e-02, -1.993330419e-01, 7.954464853e-02, -1.615219861e-01, -5.934982300e-01, -5.788858235e-02, 3.136359155e-01, 1.605119258e-01, 2.108585089e-01, -4.703115523e-01, 3.710929155e-01, -5.417601466e-01, 2.320159227e-01, 5.445683599e-01, -2.707546353e-01, 1.840245575e-01, -2.802460790e-01, -5.368059278e-01, -4.606449306e-01, -2.137972564e-01};

    constexpr std::array<Scalar, 8> biases_12 = {-1.995351315e-01, 2.065921277e-01, -6.383392960e-02, 2.270492017e-01, 1.386604756e-01, -2.522695661e-01, -9.076008201e-02, -1.786617935e-01};

    constexpr std::array<Scalar, 8> gamma_13 = {8.171702027e-01, 8.087249994e-01, 8.240170479e-01, 7.342007756e-01, 8.018008471e-01, 8.554974794e-01, 8.706863523e-01, 7.867746949e-01};

    constexpr std::array<Scalar, 8> beta_13 = {-8.743448555e-02, -1.272444241e-02, 7.456973195e-02, 6.852093339e-02, 3.641036153e-02, 3.186487034e-02, 7.128121704e-02, -1.124290079e-01};

    constexpr Scalar epsilon_13 = 1.000000000e-03;

    constexpr std::array<Scalar, 80> weights_14 = {8.528346568e-02, -2.350563854e-01, 1.217049509e-01, -8.846273273e-02, 1.256122142e-01, -1.204561368e-01, 5.605119467e-02, -1.297867894e-01, -7.027309388e-02, -8.842495829e-02, 8.730350435e-02, 1.181214452e-01, 1.584716514e-02, 2.450873554e-01, -1.424113363e-01, -1.880910695e-01, -1.676942706e-01, 6.331332773e-02, 3.412071466e-01, 1.177171841e-01, -2.227726951e-02, -1.322735399e-01, -4.942917824e-02, 1.762154698e-01, -2.631161809e-01, 2.811429799e-01, 8.144146204e-02, -7.232605815e-01, -3.292691112e-01, 7.550350577e-02, 8.810706437e-02, -1.082232967e-02, -1.337719848e-03, 3.826901913e-01, -4.817551747e-02, 1.381167471e-01, 2.130220234e-01, -3.604719043e-01, 1.864458323e-01, 1.201805025e-01, 2.007466853e-01, -1.440710276e-01, 2.407129556e-01, -2.730527148e-02, 3.594694138e-01, -3.226770088e-02, -7.031956222e-03, 2.009566315e-02, -1.730855703e-01, -2.338857576e-02, -1.439502984e-01, -5.987040699e-02, -3.328115046e-01, 2.579571605e-01, -2.911436260e-01, -2.063767016e-01, -2.481583357e-01, -3.685392737e-01, 1.667720824e-01, 6.645186990e-02, -1.059023663e-02, -1.442592591e-01, -1.740476675e-02, -4.948105663e-02, -1.173714921e-01, -4.782869220e-01, -1.183153838e-01, 2.495291233e-01, 3.648656309e-01, -7.899560034e-02, -2.566860057e-02, -1.342843026e-01, 6.431555748e-02, -6.070753559e-02, -3.543513715e-01, -1.805955768e-01, -1.798266359e-02, -2.392023653e-01, -1.096900180e-01, -1.073622555e-01};

    constexpr std::array<Scalar, 10> biases_14 = {6.280402094e-02, 1.201294661e-01, 3.610751033e-02, 6.711626053e-02, -6.421606988e-02, 8.850201219e-02, 1.209345236e-01, 8.016167581e-02, 1.000001878e-01, 1.554360688e-01};

    //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// 

    auto elu = [](Scalar& output, Scalar input, Scalar alpha) noexcept {
        output = input > 0 ? input : alpha * (std::exp(input) - 1);
    };

    auto linear = [](Scalar& output, Scalar input, Scalar alpha) noexcept {
        output = input;
    };

    auto sigmoid = [](Scalar& output, Scalar input, Scalar alpha) noexcept {
        output = 1 / (1 + std::exp(-input));
    };

    auto silu = [](Scalar& output, Scalar input, Scalar alpha) noexcept {
        auto sigmoid = 1 / (1 + std::exp(-input));
        output = input * sigmoid;
    };

    auto relu = [](Scalar& output, Scalar input, Scalar alpha) noexcept {
        output = input > 0 ? input : 0;
    };

    auto tanhCustom = [](Scalar& output, Scalar input, Scalar alpha) noexcept {
        output = std::tanh(input);
    };

    //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// 

    constexpr std::array<Scalar, 21> layer_shape = {(0, 3), 0, 0, ((3, 8), (8,)), 0, ((8, 8), (8,)), 0, ((8,), (8,), 1), ((8, 8), (8,)), 0, ((8,), (8,), (8,), (8,), 1), ((8, 8), (8,)), 0, 0, 0, 0, ((8, 8), (8,)), 0, ((8,), (8,), 1), ((8, 10), (10,)), 0};

    std::array<Scalar, 3> layer_1_output;
    relu(layer_1_output[0], model_input[0], 0.0);
    relu(layer_1_output[1], model_input[1], 0.0);
    relu(layer_1_output[2], model_input[2], 0.0);

    std::array<Scalar, 3> layer_2_output;
    linear(layer_2_output[0], layer_1_output[0], 0.0);
    linear(layer_2_output[1], layer_1_output[1], 0.0);
    linear(layer_2_output[2], layer_1_output[2], 0.0);

    std::array<Scalar, 8> layer_3_output;
    forwardPass<Scalar, 8>(layer_3_output.data(), layer_2_output.data(), weights_3.data(), biases_3.data(), 3, linear, 0.0);

    std::array<Scalar, 8> layer_4_output;
    forwardPass<Scalar, 8>(layer_4_output.data(), layer_3_output.data(), weights_4.data(), biases_4.data(), 8, silu, 0.0);

    std::array<Scalar, 8> layer_5_output;
    layerNormalization<Scalar, 8>(layer_5_output.data(), layer_4_output.data(), gamma_5.data(), beta_5.data(), epsilon_5);

    std::array<Scalar, 8> layer_6_output;
    forwardPass<Scalar, 8>(layer_6_output.data(), layer_5_output.data(), weights_6.data(), biases_6.data(), 8, tanhCustom, 0.0);

    std::array<Scalar, 8> layer_7_output;
    batchNormalization<Scalar, 8>(layer_7_output.data(), layer_6_output.data(), gamma_7.data(), beta_7.data(), mean_7.data(), variance_7.data(), epsilon_7);

    std::array<Scalar, 8> layer_8_output;
    forwardPass<Scalar, 8>(layer_8_output.data(), layer_7_output.data(), weights_8.data(), biases_8.data(), 8, linear, 0.0);

    std::array<Scalar, 8> layer_9_output;
    sigmoid(layer_9_output[0], layer_8_output[0], 0.0);
    sigmoid(layer_9_output[1], layer_8_output[1], 0.0);
    sigmoid(layer_9_output[2], layer_8_output[2], 0.0);
    sigmoid(layer_9_output[3], layer_8_output[3], 0.0);
    sigmoid(layer_9_output[4], layer_8_output[4], 0.0);
    sigmoid(layer_9_output[5], layer_8_output[5], 0.0);
    sigmoid(layer_9_output[6], layer_8_output[6], 0.0);
    sigmoid(layer_9_output[7], layer_8_output[7], 0.0);

    std::array<Scalar, 8> layer_10_output;
    linear(layer_10_output[0], layer_9_output[0], 0.0);
    linear(layer_10_output[1], layer_9_output[1], 0.0);
    linear(layer_10_output[2], layer_9_output[2], 0.0);
    linear(layer_10_output[3], layer_9_output[3], 0.0);
    linear(layer_10_output[4], layer_9_output[4], 0.0);
    linear(layer_10_output[5], layer_9_output[5], 0.0);
    linear(layer_10_output[6], layer_9_output[6], 0.0);
    linear(layer_10_output[7], layer_9_output[7], 0.0);

    std::array<Scalar, 8> layer_11_output;
    elu(layer_11_output[0], layer_10_output[0], 1.0);
    elu(layer_11_output[1], layer_10_output[1], 1.0);
    elu(layer_11_output[2], layer_10_output[2], 1.0);
    elu(layer_11_output[3], layer_10_output[3], 1.0);
    elu(layer_11_output[4], layer_10_output[4], 1.0);
    elu(layer_11_output[5], layer_10_output[5], 1.0);
    elu(layer_11_output[6], layer_10_output[6], 1.0);
    elu(layer_11_output[7], layer_10_output[7], 1.0);

    std::array<Scalar, 8> layer_12_output;
    forwardPass<Scalar, 8>(layer_12_output.data(), layer_11_output.data(), weights_12.data(), biases_12.data(), 8, linear, 0.0);

    std::array<Scalar, 8> layer_13_output;
    layerNormalization<Scalar, 8>(layer_13_output.data(), layer_12_output.data(), gamma_13.data(), beta_13.data(), epsilon_13);

    std::array<Scalar, 10> layer_14_output;
    forwardPass<Scalar, 10>(layer_14_output.data(), layer_13_output.data(), weights_14.data(), biases_14.data(), 8, linear, 0.0);

    constexpr std::array<Scalar, 10> output_norms = {9.934309616e-01, 9.617949734e-01, 9.796113737e-01, 9.758307726e-01, 9.646616886e-01, 9.946274980e-01, 9.858196838e-01, 9.538358071e-01, 9.862610589e-01, 9.814134212e-01};

    constexpr std::array<Scalar, 10> output_mins = {1.280830518e-03, 3.093672333e-02, 1.008834337e-02, 1.829334318e-02, 1.811821693e-02, 3.327897599e-03, 1.416018130e-02, 1.323092537e-02, 2.370498897e-03, 5.197589451e-03};

    std::array<Scalar, 10> model_output;
    for (int i = 0; i < 10; i++) { model_output[i] = (layer_14_output[i] * output_norms[i]) + output_mins[i]; }
    return model_output;
}
