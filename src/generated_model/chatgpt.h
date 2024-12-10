#include <iostream>
#include <array>
#include <random>
#include <cmath>
#include <stdexcept>
#include <algorithm> // for std::max

template<typename Scalar>
using activationFunction = void(*)(Scalar* __restrict, const Scalar* __restrict, size_t, Scalar);

// - -

template<typename Scalar>
void relu(Scalar* __restrict outputs, const Scalar* __restrict inputs, size_t size, Scalar alpha = 0.0) noexcept {
    #pragma omp simd
    for (size_t i = 0; i < size; ++i) {
        outputs[i] = std::max(inputs[i], Scalar(0));
    }
}

template<typename Scalar, int size>
void batchNormalization(Scalar* __restrict outputs, const Scalar* __restrict inputs,
                        const Scalar* __restrict gamma, const Scalar* __restrict beta,
                        const Scalar* __restrict mean, const Scalar* __restrict variance,
                        const Scalar epsilon) noexcept {
    #pragma omp simd
    for (int i = 0; i < size; ++i) {
        outputs[i] = gamma[i] * ((inputs[i] - mean[i]) / std::sqrt(variance[i] + epsilon)) + beta[i];
    }
}

template<typename Scalar, int size>
void layerNormalization(Scalar* __restrict outputs, const Scalar* __restrict inputs,
                        const Scalar* __restrict gamma, const Scalar* __restrict beta,
                        Scalar epsilon) noexcept {
    Scalar mean = 0;
    #pragma omp simd reduction(+:mean)
    for (int i = 0; i < size; ++i) {
        mean += inputs[i];
    }
    mean /= size;

    Scalar variance = 0;
    #pragma omp simd reduction(+:variance)
    for (int i = 0; i < size; ++i) {
        Scalar diff = inputs[i] - mean;
        variance += diff * diff;
    }
    variance /= size;

    Scalar denom = Scalar(1) / std::sqrt(variance + epsilon);
    #pragma omp simd
    for (int i = 0; i < size; ++i) {
        outputs[i] = gamma[i] * ((inputs[i] - mean) * denom) + beta[i];
    }
}

template<typename Scalar>
void linear(Scalar* __restrict outputs, const Scalar* __restrict inputs, size_t size, Scalar alpha = 0.0) noexcept {
    #pragma omp simd
    for (size_t i = 0; i < size; ++i) {
        outputs[i] = inputs[i];
    }
}

template<typename Scalar>
void dotProduct(Scalar* __restrict outputs, const Scalar* __restrict inputs,
                const Scalar* __restrict weights, int input_size, int output_size) noexcept {
    for (int i = 0; i < output_size; i++) {
        Scalar sum = Scalar(0);
        #pragma omp simd reduction(+:sum)
        for (int j = 0; j < input_size; j++) {
            sum += inputs[j] * weights[j * output_size + i];
        }
        outputs[i] = sum;
    }
}

template<typename Scalar>
void addBias(Scalar* __restrict outputs, const Scalar* __restrict biases, int size) noexcept {
    #pragma omp simd
    for (int i = 0; i < size; i++) {
        outputs[i] += biases[i];
    }
}

template<typename Scalar, int output_size>
void forwardPass(Scalar* __restrict outputs, const Scalar* __restrict inputs,
                 const Scalar* __restrict weights, const Scalar* __restrict biases,
                 int input_size,
                 void (*activation_function)(Scalar* __restrict, const Scalar* __restrict, size_t, Scalar),
                 Scalar alpha) noexcept {
    std::array<Scalar, output_size> temp_outputs;
    dotProduct(temp_outputs.data(), inputs, weights, input_size, output_size);
    addBias(temp_outputs.data(), biases, output_size);
    activation_function(outputs, temp_outputs.data(), output_size, alpha);
}

// - -

template <typename Scalar = double>
auto chatgpt(const std::array<Scalar, 3>& initial_input) {
    std::array<Scalar, 3> model_input = initial_input;

    if (model_input.size() != 3) {
        throw std::invalid_argument("Invalid input size. Expected size: 3");
    }

    // - -

    std::array<Scalar, 24> weights_1 = {
        4.372317195e-01, -6.456755996e-01, -5.946626067e-01, 1.697454154e-01,
        -5.854411721e-01, -7.242655754e-01, 5.806417465e-01, -3.160373271e-01,
        -7.321768403e-01, 2.998109460e-01, 7.293194532e-02, 1.261655390e-01,
        4.299084842e-01, 2.532907724e-01, -6.458786130e-01, 6.060208678e-01,
        -1.317059100e-01, 3.070001602e-01, 6.017443538e-01, 8.147881031e-01,
        3.523627818e-01, 4.023167491e-01, -4.509314001e-01, -2.286254913e-01
    };

    std::array<Scalar, 8> biases_1 = {
        4.805250093e-02, 7.251290977e-02, -1.182154845e-02, -1.616869867e-02,
        -7.863835245e-02, -1.885187626e-02, 9.885193780e-03, -1.522710640e-02
    };

    std::array<Scalar, 8> gamma_2 = {
        8.530170918e-01, 8.962954283e-01, 1.087913394e+00, 1.161609173e+00,
        1.011281371e+00, 1.041467547e+00, 9.014381170e-01, 8.387947679e-01
    };

    std::array<Scalar, 8> beta_2 = {
        -6.278064102e-02, -2.818236873e-02, -5.814415589e-02, 6.393481791e-02,
        9.988965094e-02, 7.637216896e-02, -1.175570041e-01, -4.125739262e-02
    };

    std::array<Scalar, 8> mean_2 = {
        3.961207345e-02, 1.112300828e-01, 1.092181727e-01, 5.711241961e-01,
        1.040857285e-01, 7.434368134e-02, 2.986635268e-02, 8.445216715e-02
    };

    std::array<Scalar, 8> variance_2 = {
        6.365191657e-03, 1.977518573e-02, 2.148102969e-02, 5.693700165e-02,
        1.895120367e-02, 1.426414773e-02, 5.339279305e-03, 1.206206996e-02
    };

    Scalar epsilon_2 = 1.000000000e-03;

    std::array<Scalar, 128> weights_3 = {
        -1.602814496e-01, -3.741484880e-01,  3.484163284e-01,  1.449507922e-01,
        -3.050493896e-01, -3.972343504e-01, -2.485997975e-02,  2.228429765e-01,
         5.911922455e-02, -3.297399282e-01,  1.564459354e-01,  9.573768079e-02,
         6.782162189e-02, -3.399696946e-01,  3.940840662e-01, -2.667054534e-01,
         3.520442247e-01, -5.790857226e-02,  4.313700497e-01, -1.434709728e-01,
         3.351948410e-02, -2.471738309e-01, -3.687763810e-01,  4.361280203e-01,
        -3.085174263e-01,  2.333767116e-01, -3.558333516e-01, -1.666327119e-01,
         2.015216649e-01, -5.661841109e-02, -7.083129883e-02,  9.174793214e-02,
         6.964109093e-02, -5.873147249e-01,  3.068975210e-01, -2.824039459e-01,
        -4.499334395e-01,  4.402310550e-01,  2.369143665e-01, -9.100596607e-02,
        -9.137083590e-02, -3.219597638e-01,  2.701722980e-01, -4.983503371e-02,
        -3.905560970e-01, -1.482723206e-01,  1.965867877e-01,  1.503220797e-01,
        -2.299677581e-02, -4.559743404e-01, -3.223949373e-01,  5.811718702e-01,
        -3.871290982e-01,  3.096524775e-01, -4.628811479e-01,  6.223252974e-03,
         5.460524559e-01, -1.226334646e-01, -1.696113944e-01, -2.625229061e-01,
        -8.961799741e-02,  2.975621819e-01, -3.901964426e-01, -3.800031245e-01,
        -3.528463840e-01, -3.749357164e-01, -4.601852298e-01,  2.406878769e-01,
        -1.174056381e-01,  3.382272422e-01, -1.501182020e-01, -4.757212996e-01,
         1.730065793e-01, -3.385700285e-01, -3.246173561e-01, -1.986623257e-01,
        -1.144706309e-01, -5.152844638e-02, -1.976905763e-02, -3.462117165e-02,
         2.914912701e-01,  4.366855919e-01,  2.902888656e-01,  1.411571950e-01,
        -3.481039703e-01,  3.704413772e-01, -4.055422246e-01, -1.302174479e-01,
        -2.699041069e-01,  5.466046929e-01,  1.586918235e-01,  1.979544312e-01,
        -3.618941009e-01,  2.397110872e-02,  7.480193675e-02, -4.521637857e-01,
        -2.348930985e-01, -3.290480971e-01,  1.490076631e-01,  1.496319622e-01,
        -4.642343521e-01,  3.601624966e-01,  1.901465803e-01,  7.593458518e-03,
        -5.733768344e-01,  1.419606507e-01, -1.583343148e-01,  6.619928032e-02,
        -2.852062583e-01,  8.897942491e-03,  3.219449520e-01,  3.153575361e-01,
        -3.474158347e-01, -3.870499134e-01, -2.541141212e-01,  3.468716741e-01,
         4.630386829e-01,  1.465570480e-01, -2.172984779e-01,  2.535189986e-01,
        -7.728642970e-02,  4.071654081e-01,  1.390026510e-01, -5.265063643e-01,
        -4.025568366e-01, -1.486774832e-01, -1.823622584e-01,  5.802583322e-02
    };

    std::array<Scalar, 16> biases_3 = {
        3.760609403e-02,  9.721727669e-02,  8.817947656e-02,  8.211687207e-02,
        8.665796369e-02,  7.620748132e-02, -4.674670845e-02, -1.235103533e-01,
        2.890625894e-01, -3.376184404e-02, -8.472009003e-02, -4.444331676e-02,
        -3.610830754e-02, -1.519946456e-01, -3.825052083e-02, -2.453378215e-02
    };

    std::array<Scalar, 16> gamma_4 = {
        9.552657008e-01, 9.403786063e-01, 9.198512435e-01, 1.074991465e+00,
        1.039478540e+00, 9.564712048e-01, 8.727928400e-01, 1.031798720e+00,
        1.214746237e+00, 8.237137794e-01, 9.796001315e-01, 9.356305003e-01,
        1.096121669e+00, 1.161767125e+00, 1.017645478e+00, 9.581672549e-01
    };

    std::array<Scalar, 16> beta_4 = {
        -1.427260339e-01,  1.817565113e-01,  2.046239674e-01, -7.580352575e-02,
         6.024412811e-02,  7.853285223e-02,  5.804245546e-02,  3.294105828e-02,
         2.247500420e-01, -1.979366839e-01, -2.541813552e-01,  7.864131033e-02,
        -1.597461104e-01, -8.799260110e-02, -9.535875171e-03, -2.231868543e-02
    };

    Scalar epsilon_4 = 1.000000000e-03;

    std::array<Scalar, 128> weights_5 = {
        -4.319393337e-01, -4.576797783e-01, -2.125814110e-01, -6.122328341e-02,
         9.470989555e-02,  4.424705729e-02,  2.020039409e-01,  1.975794286e-01,
         3.580464721e-01,  4.482282400e-01, -3.832866624e-02, -4.509196877e-01,
         2.379722297e-01, -1.873977855e-02,  3.889089525e-01, -2.623267770e-01,
         2.386791408e-01,  1.338053644e-01,  3.364596367e-01, -2.924877703e-01,
         3.643762469e-01, -1.639985740e-01,  3.743162155e-01,  9.347656369e-02,
        -4.624824524e-01, -5.727112889e-01, -1.615149379e-01, -2.798598111e-01,
         2.267042249e-01, -3.737767935e-01,  1.425231695e-01,  2.131708562e-01,
         3.125389814e-01,  1.556640416e-01,  2.759712040e-01, -2.055747211e-01,
        -2.674479485e-01,  2.547337115e-01, -4.283007085e-01, -1.796641573e-02,
        -9.640584141e-02, -3.502503335e-01, -2.893632054e-01, -4.602450728e-01,
         4.560279846e-01, -2.409231365e-01,  2.269199491e-01,  1.643139496e-02,
         2.884392142e-01, -2.887033522e-01,  5.477294326e-01, -2.902758718e-01,
        -2.012702823e-01, -1.487420592e-02,  3.802637756e-01, -4.496557713e-01,
         5.019686818e-01, -5.501865745e-01,  2.046861053e-01, -3.915296495e-01,
        -3.911296725e-01, -4.731472731e-01,  1.475513726e-01, -3.400562406e-01,
         6.167630851e-02, -2.191629410e-01, -4.723451287e-02, -1.540697068e-01,
         4.746681750e-01,  3.395751864e-02,  4.939040244e-01, -1.141399741e-01,
         1.347191632e-01, -7.294050604e-02,  3.489517570e-01, -4.320429265e-02,
        -3.717319667e-01,  2.749838531e-01, -5.052853823e-01, -8.071258664e-02,
        -9.953162074e-02, -1.493071578e-02, -2.570554614e-01,  9.063125402e-02,
        -9.541187435e-02, -9.308911860e-03, -3.704628348e-01, -4.891365170e-01,
         3.134778440e-01,  5.062409043e-01,  1.348799560e-02, -4.298295379e-01,
        -4.874583185e-01, -3.008376360e-01, -4.101943374e-01, -3.829127178e-02,
        -3.201882541e-01, -4.975683391e-01, -1.781423986e-01,  9.011033922e-02,
         4.223510027e-01,  1.091246027e-02,  2.959837671e-03,  3.064734340e-01,
        -3.063872457e-01, -5.583360195e-01, -6.162431836e-01, -2.535644770e-01,
        -3.000959456e-01, -6.584655643e-01,  1.051133275e-01,  6.520739291e-03,
         3.038018048e-01,  2.586948276e-01, -1.347907186e-01,  2.868345082e-01,
        -1.929909438e-01,  4.559011459e-01, -2.615417838e-01, -2.471874803e-01,
         4.346833006e-02, -2.032401264e-01, -6.067797542e-02,  1.631595790e-01,
         7.634198666e-02,  3.131833673e-01,  2.354695052e-01, -2.488459051e-01
    };

    std::array<Scalar, 8> biases_5 = {
        2.582834363e-01, 2.311471701e-01, 5.219763145e-02, -7.794558257e-02,
        1.256980002e-01, 1.225228831e-01, 3.299221098e-01, 1.675228626e-01
    };

    std::array<Scalar, 8> gamma_6 = {
        8.366626501e-01, 7.690058947e-01, 8.281601667e-01, 6.870509386e-01,
        8.429152966e-01, 8.444930315e-01, 7.407457232e-01, 7.318800688e-01
    };

    std::array<Scalar, 8> beta_6 = {
        -2.179592997e-01,  1.542007774e-01,  2.112130076e-01,  2.272527069e-01,
         2.322434038e-01, -2.508309186e-01, -2.875189185e-01, -2.541064024e-01
    };

    std::array<Scalar, 8> mean_6 = {
        1.126821876e+00, 1.416609526e+00, 9.007228017e-01, 5.064993352e-02,
        1.601657510e+00, 8.773842454e-01, 1.473725080e+00, 5.832844973e-01
    };

    std::array<Scalar, 8> variance_6 = {
        1.224569678e+00, 1.854018807e+00, 8.017200828e-01, 2.375720814e-02,
        1.775315642e+00, 8.018508554e-01, 1.490316153e+00, 2.816684544e-01
    };

    Scalar epsilon_6 = 1.000000000e-03;

    std::array<Scalar, 80> weights_7 = {
        -2.327001542e-01, -5.443356000e-03, -3.732218444e-01, -1.113607362e-01,
        -1.038733199e-01,  2.722984180e-02, -4.754949808e-01, -4.982326925e-01,
        -4.434935749e-01, -4.688684046e-01,  1.606160253e-01,  7.354979217e-02,
         2.352750450e-01, -1.147493124e-01,  2.387378365e-02,  1.151713282e-01,
         2.049409896e-01,  2.248842120e-01,  1.907670349e-01,  1.045567319e-01,
         2.039306611e-01,  4.494054615e-01,  3.557773232e-01,  3.321146369e-01,
        -1.970963627e-01,  3.223553002e-01, -2.644304633e-01,  5.414409041e-01,
         2.730385661e-01,  2.331824303e-01,  4.627565295e-02,  1.375712901e-01,
         4.735491052e-02,  3.302559257e-02,  1.655320870e-03, -2.481907792e-02,
         7.448184490e-02,  5.502271652e-02,  9.191073477e-03,  8.935330063e-02,
         3.036630750e-01,  4.494018555e-01,  1.910171956e-01,  5.729774479e-03,
         4.747701436e-02,  3.705445230e-01,  1.649201661e-02,  2.303286642e-01,
        -3.207141533e-02,  1.748633236e-01, -7.711198181e-02,  1.936862431e-02,
        -7.862249017e-02, -2.371163219e-01, -3.261410445e-02, -3.242918253e-01,
         7.388048619e-02, -1.304937154e-01, -2.834713161e-01,  1.861578971e-01,
        -1.154974326e-01, -8.241957985e-03, -1.585841924e-01,  4.265726358e-02,
        -2.154877782e-01, -4.062551856e-01, -2.168684006e-01, -8.542352915e-02,
        -1.335958242e-01, -1.105253175e-01, -2.072376162e-01,  1.653679013e-01,
         1.258789748e-01, -2.202370763e-01, -2.350620478e-01,  8.778560162e-02,
        -3.564902544e-01, -6.356771290e-02, -1.954660714e-01, -5.796959624e-02
    };

    std::array<Scalar, 10> biases_7 = {
        1.996348500e-01, 3.428412676e-01, 1.965957731e-01, 3.577214181e-01,
        3.781837225e-01, 2.174095809e-01, 2.556671798e-01, 7.218646258e-02,
        1.112452820e-01, 2.370930612e-01
    };

    // Forward pass
    std::array<Scalar, 8> layer_1_output;
    forwardPass<Scalar, 8>(layer_1_output.data(), model_input.data(), weights_1.data(), biases_1.data(), 3, &relu<Scalar>, Scalar(0));

    std::array<Scalar, 8> layer_2_output;
    batchNormalization<Scalar, 8>(layer_2_output.data(), layer_1_output.data(), gamma_2.data(), beta_2.data(), mean_2.data(), variance_2.data(), epsilon_2);

    std::array<Scalar, 16> layer_3_output;
    forwardPass<Scalar, 16>(layer_3_output.data(), layer_2_output.data(), weights_3.data(), biases_3.data(), 8, &relu<Scalar>, Scalar(0));

    std::array<Scalar, 16> layer_4_output;
    layerNormalization<Scalar, 16>(layer_4_output.data(), layer_3_output.data(), gamma_4.data(), beta_4.data(), epsilon_4);

    std::array<Scalar, 8> layer_5_output;
    forwardPass<Scalar, 8>(layer_5_output.data(), layer_4_output.data(), weights_5.data(), biases_5.data(), 16, &relu<Scalar>, Scalar(0));

    std::array<Scalar, 16> layer_6_output;
    batchNormalization<Scalar, 16>(layer_6_output.data(), layer_5_output.data(), gamma_6.data(), beta_6.data(), mean_6.data(), variance_6.data(), epsilon_6);

    std::array<Scalar, 10> layer_7_output;
    forwardPass<Scalar, 10>(layer_7_output.data(), layer_6_output.data(), weights_7.data(), biases_7.data(), 16, &linear<Scalar>, Scalar(0));

    return layer_7_output;
}