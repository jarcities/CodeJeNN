// #include <iostream>
// #include <array>
// #include <random>
// #include <cmath>
// #include <utility>

// template<typename Scalar>
// using activationFunction=void(*)(Scalar&,Scalar,Scalar);

// template<typename Scalar> constexpr void relu(Scalar&o,Scalar i,Scalar a=0.0)noexcept{o=i>0?i:0;}
// template<typename Scalar> constexpr void linear(Scalar&o,Scalar i,Scalar a=0.0)noexcept{o=i;}
// template<typename Scalar> constexpr void silu(Scalar&o,Scalar i,Scalar a=1.0)noexcept{Scalar s=1/(1+std::exp(-i));o=i*s;}
// template<typename Scalar> constexpr void tanhCustom(Scalar&o,Scalar i,Scalar a=0.0)noexcept{o=std::tanh(i);}
// template<typename Scalar> constexpr void sigmoid(Scalar&o,Scalar i,Scalar a=0.0)noexcept{o=1/(1+std::exp(-i));}
// template<typename Scalar> constexpr void elu(Scalar&o,Scalar i,Scalar a)noexcept{o=i>0?i:a*(std::exp(i)-1);}
// template<typename Scalar> constexpr void addBias(Scalar&o,Scalar b)noexcept{o+=b;}
// template<typename Scalar> constexpr void dotProduct(Scalar&s,Scalar in,Scalar w)noexcept{s+=in*w;}

// template<typename Scalar,int size,std::size_t...Is>
// inline Scalar computeSum(const Scalar* in,std::index_sequence<Is...>)noexcept{return((in[Is]+...));}

// template<typename Scalar,int size,std::size_t...Is>
// inline Scalar computeSumSquares(const Scalar* in,Scalar m,std::index_sequence<Is...>)noexcept{return(((in[Is]-m)*(in[Is]-m))+...);}

// template<typename Scalar,int size,std::size_t...Is>
// inline void layerNormalizationImpl(Scalar* o,const Scalar* in,const Scalar* g,const Scalar* b,Scalar M,Scalar V,Scalar e,std::index_sequence<Is...>)noexcept{((o[Is]=g[Is]*((in[Is]-M)/std::sqrt(V+e))+b[Is]),...);}

// template<typename Scalar,int size>
// inline void layerNormalization(Scalar* o,const Scalar* in,const Scalar* g,const Scalar* B,Scalar e)noexcept{
//     Scalar M=computeSum<Scalar,size>(in,std::make_index_sequence<size>{})/static_cast<Scalar>(size);
//     Scalar V=computeSumSquares<Scalar,size>(in,M,std::make_index_sequence<size>{})/static_cast<Scalar>(size);
//     layerNormalizationImpl<Scalar,size>(o,in,g,B,M,V,e,std::make_index_sequence<size>{});
// }

// template<typename Scalar,int size,std::size_t...Is>
// inline void batchNormalizationImpl(Scalar* o,const Scalar* in,const Scalar* G,const Scalar* B,const Scalar* M,const Scalar* V,Scalar e,std::index_sequence<Is...>)noexcept{((o[Is]=G[Is]*((in[Is]-M[Is])/std::sqrt(V[Is]+e))+B[Is]),...);}

// template<typename Scalar,int size>
// inline void batchNormalization(Scalar* o,const Scalar* in,const Scalar* G,const Scalar* B,const Scalar* M,const Scalar* V,Scalar e)noexcept{
//     batchNormalizationImpl<Scalar,size>(o,in,G,B,M,V,e,std::make_index_sequence<size>{});
// }

// template<typename Scalar,int output_size,int input_size,std::size_t...Is>
// inline void forwardPassImpl(Scalar* o,const Scalar* in,const std::array<std::array<Scalar,output_size>,input_size>& w,const Scalar* b,activationFunction<Scalar> f,Scalar a,std::index_sequence<Is...>)noexcept{
//     (([&]{Scalar S=0;
// #pragma clang loop vectorize(enable)
// #pragma GCC ivdep
// for(int j=0;j<input_size;++j)S+=in[j]*w[j][Is];S+=b[Is];f(o[Is],S,a);}()),...);
// }

// template<typename Scalar,int output_size,int input_size>
// inline void forwardPass(Scalar* o,const Scalar* in,const std::array<std::array<Scalar,output_size>,input_size>& w,const std::array<Scalar,output_size>& b,activationFunction<Scalar> f,Scalar a)noexcept{
//     forwardPassImpl<Scalar,output_size,input_size>(o,in,w,b.data(),f,a,std::make_index_sequence<output_size>{});
// }

// template<typename Scalar,std::size_t Size,std::size_t...Is>
// inline void applyActivationFunctionsImpl(std::array<Scalar,Size>& o,const std::array<Scalar,Size>& in,activationFunction<Scalar> f,Scalar a,std::index_sequence<Is...>)noexcept{(f(o[Is],in[Is],a),...);}

// template<typename Scalar,std::size_t Size>
// inline void applyActivationFunctions(std::array<Scalar,Size>& o,const std::array<Scalar,Size>& in,activationFunction<Scalar> f,Scalar a)noexcept{
//     applyActivationFunctionsImpl<Scalar,Size>(o,in,f,a,std::make_index_sequence<Size>{});
// }

// // --

// template<typename Scalar=double>
// auto newArray(const std::array<Scalar,3>& initial_input){
//     std::array<Scalar,3> model_input=initial_input;
//     std::array<std::array<Scalar,8>,3> weights_3={{ {1.326323599e-01,-4.964405894e-01,-4.476362765e-01,4.825202525e-01,4.999661446e-01,6.104987264e-01,2.875608206e-01,-3.064861000e-01},
//                                                      {3.347294331e-01,2.181584835e-01,-3.532409966e-01,6.674022079e-01,-6.745031476e-01,3.237143457e-01,-2.341475338e-01,-5.993123055e-01},
//                                                      {1.393576264e-01,3.206691891e-02,-4.254285693e-01,6.793647408e-01,-6.585419774e-01,-6.157318950e-01,1.563780308e-01,-2.918911166e-02}}};
//     std::array<Scalar,8> biases_3={-9.336775169e-03,1.460986445e-03,1.543075219e-02,2.364202775e-02,-4.833743349e-02,4.356297478e-02,3.732505068e-02,-4.034891352e-02};
//     std::array<std::array<Scalar,8>,8> weights_4={{ {2.895489633e-01,1.404272467e-01,-5.511720181e-01,-1.963775754e-01,-4.318673015e-01,-6.175366044e-01,3.609300554e-01,-4.942331910e-01},
//                                                      {1.385588199e-01,5.399128795e-01,-4.022693336e-01,4.465122521e-01,7.421918213e-02,4.528645426e-02,-2.302111685e-01,3.943967819e-01},
//                                                      {-3.330304623e-01,9.097249061e-02,-3.297064081e-02,6.087180972e-01,-2.183382064e-01,-8.720761538e-02,5.392811298e-01,-3.553541601e-01},
//                                                      {-4.102258384e-01,-3.817836344e-01,-3.408961594e-01,-5.070082545e-01,-9.703770280e-02,1.385468096e-01,-3.676263988e-01,1.009406447e-01},
//                                                      {-7.793208212e-02,-2.422472984e-01,2.013003081e-01,4.458051324e-01,-6.043104529e-01,-1.048356891e-01,-1.337849349e-01,3.964666724e-01},
//                                                      {-5.302410126e-01,-1.840705723e-01,-4.131732583e-01,4.750281870e-01,5.418121815e-01,-4.915194511e-01,2.186830342e-01,-5.351045132e-01},
//                                                      {-5.902838111e-01,4.853354022e-02,-2.235947996e-01,3.651028574e-01,8.357726038e-02,-8.591035753e-02,1.432061195e-01,-2.278156132e-01},
//                                                      {5.197193027e-01,-3.674283326e-01,4.082169831e-01,-1.086276844e-01,1.694692671e-01,1.661305875e-01,-1.906166077e-01,-4.951056540e-01}}};
//     std::array<Scalar,8> biases_4={-4.018265381e-02,7.817581296e-03,-2.712406032e-02,-2.441039495e-02,4.375640303e-02,2.640516497e-03,4.310625419e-02,2.533612400e-02};
//     std::array<Scalar,8> gamma_5={1.007159472e+00,9.102494121e-01,9.754046202e-01,1.009639382e+00,9.939744473e-01,1.053939223e+00,1.005302906e+00,1.049999237e+00};
//     std::array<Scalar,8> beta_5={-6.327476352e-02,1.301053464e-01,-8.200552315e-02,3.901715577e-02,1.436686069e-01,1.086948216e-01,9.198349435e-04,1.006997824e-01};
//     Scalar epsilon_5=1.000000000e-03;
//     std::array<std::array<Scalar,8>,8> weights_6={{ {-4.491657019e-01,1.051441580e-02,-8.537553996e-02,-4.117136896e-01,5.043860078e-01,6.056678295e-01,1.615756936e-02,-1.299536973e-01},
//                                                      {3.373832703e-01,2.744476497e-01,2.461240739e-01,5.393412113e-01,-2.386006713e-01,-2.250907719e-01,4.689204097e-01,2.077780962e-01},
//                                                      {-1.299372762e-01,-3.652926087e-01,1.432644129e-01,2.902889252e-01,-3.890721798e-01,-4.642286897e-01,-5.835254192e-01,-1.329570264e-01},
//                                                      {5.750157312e-02,2.351277173e-01,8.795207739e-02,4.416449368e-01,4.325707555e-01,-4.251926541e-01,4.364499748e-01,-6.390288472e-01},
//                                                      {5.611011386e-01,3.921681643e-01,3.791545928e-01,-2.027980834e-01,-1.832287610e-01,5.065606236e-01,5.255874395e-01,5.944129229e-01},
//                                                      {4.012452364e-01,4.532038048e-02,-2.581494451e-01,-4.223471582e-01,-2.514020205e-01,4.273546338e-01,5.800563097e-01,4.546979666e-01},
//                                                      {2.222973853e-01,1.506377310e-01,-9.531598538e-03,1.103633493e-01,-4.755297899e-01,4.269180000e-01,-4.701362550e-01,-5.456539392e-01},
//                                                      {6.130970716e-01,1.676640958e-01,4.789482951e-01,-2.503974438e-01,-5.631949306e-01,3.413064778e-01,-4.982459545e-01,2.986344397e-01}}};
//     std::array<Scalar,8> biases_6={1.817649454e-01,1.338390857e-01,9.126558900e-02,2.131122723e-02,2.393466048e-02,-1.300014835e-02,3.099955246e-02,6.701175869e-02};
//     std::array<Scalar,8> gamma_7={8.193984032e-01,7.379534245e-01,9.682070017e-01,9.788198471e-01,9.873768091e-01,1.000282049e+00,9.579557776e-01,7.910596132e-01};
//     std::array<Scalar,8> beta_7={2.016024888e-01,-3.292292058e-01,3.209293187e-01,-3.260774910e-01,-1.682267338e-01,-1.893343627e-01,4.644928277e-01,-3.859341145e-01};
//     std::array<Scalar,8> mean_7={9.823068976e-01,8.114451170e-01,7.645511031e-01,-6.787924170e-01,-8.765227199e-01,9.079492688e-01,6.484096050e-01,9.349939227e-01};
//     std::array<Scalar,8> variance_7={1.048183767e-04,9.622686543e-03,1.916389167e-02,8.062008768e-02,2.308205934e-03,1.166001894e-02,6.997291744e-02,2.667167410e-02};
//     Scalar epsilon_7=1.000000000e-03;
//     std::array<std::array<Scalar,8>,8> weights_8={{ {-3.972038925e-01,-3.957073092e-01,-4.971534509e-05,-2.119336426e-01,-2.219672799e-01,3.227109313e-01,4.034543037e-01,3.877579272e-01},
//                                                      {-2.669981420e-01,6.422920227e-01,-4.499044120e-01,1.564588845e-01,4.371322393e-01,4.035770893e-01,-2.477465421e-01,5.434113741e-01},
//                                                      {6.102508307e-01,-6.751357317e-01,6.106441021e-01,-5.613425374e-01,2.262963206e-01,-2.247276157e-01,-8.041384816e-02,-1.671904474e-01},
//                                                      {-4.067449868e-01,4.385202229e-01,1.891514361e-01,5.547617078e-01,2.592392862e-01,1.067967806e-02,2.351896018e-01,6.473272443e-01},
//                                                      {-3.566615880e-01,-1.914600283e-01,-4.918406308e-01,1.386630833e-01,2.280944884e-01,5.265087485e-01,4.492689371e-01,-1.003818884e-01},
//                                                      {-5.159434080e-01,-4.150543362e-02,2.560436428e-01,-4.054009914e-01,-2.460166346e-03,-5.730556846e-01,-7.491554320e-02,3.026012778e-01},
//                                                      {-1.668728143e-01,-7.224540114e-01,5.997641012e-02,-2.716617882e-01,-1.031019539e-02,9.698459506e-02,-1.572506502e-02,-6.230728030e-01},
//                                                      {-5.739236474e-01,5.079915524e-01,-4.854255319e-01,-7.777107507e-02,2.532823980e-01,3.523012996e-01,1.030851528e-01,3.888264596e-01}}};
//     std::array<Scalar,8> biases_8={2.145548761e-01,-4.947406054e-01,2.585793734e-01,-1.664437503e-01,-2.016923428e-01,2.338735759e-01,-1.787156314e-01,-1.901274323e-01};
//     std::array<std::array<Scalar,8>,8> weights_12={{ {-2.953823507e-01,9.143621475e-02,2.237670869e-01,-8.442353606e-01,6.365125179e-01,3.693207204e-01,-2.576244175e-01,-4.989452660e-01},
//                                                       {4.503370821e-01,5.406127125e-02,5.430755764e-02,4.114288390e-01,-2.794103026e-01,-5.320529938e-01,-1.550755501e-01,3.113644421e-01},
//                                                       {2.033043206e-01,4.824339449e-01,1.725191772e-01,-1.260260493e-01,7.400895357e-01,2.046696246e-01,-5.162982345e-01,-2.446578145e-01},
//                                                       {2.942177653e-01,3.172022402e-01,-1.622629352e-02,-2.305077612e-01,-1.179132536e-01,-7.849524170e-02,3.792388737e-01,-2.122235298e-01},
//                                                       {6.421005726e-01,4.533714652e-01,1.615806818e-01,1.940571330e-02,-7.184942812e-02,-2.896823883e-01,-1.747096479e-01,-2.893691361e-01},
//                                                       {-1.758630760e-02,3.972255588e-01,3.065083176e-02,-7.126296759e-01,5.962981582e-01,1.485264003e-01,-4.742491543e-01,-6.893997788e-01},
//                                                       {3.206037879e-01,2.579070926e-01,2.029581964e-01,-3.493468761e-01,5.849143490e-02,-1.007192209e-01,-4.437026102e-03,-4.000317752e-01},
//                                                       {-5.845316499e-02,1.884575784e-01,-6.055310741e-02,-3.555367887e-01,1.380394958e-02,5.944981799e-02,2.185856849e-01,-2.417995185e-01}}};
//     std::array<Scalar,8> biases_12={1.869939864e-01,1.105956733e-01,1.053466052e-01,-3.331353962e-01,1.100314036e-01,-6.662753224e-02,-9.369239211e-03,-2.431712747e-01};
//     std::array<Scalar,8> gamma_13={7.639720440e-01,8.527194262e-01,6.928566098e-01,8.144926429e-01,9.072342515e-01,9.955703616e-01,8.884121776e-01,8.234179020e-01};
//     std::array<Scalar,8> beta_13={2.525599673e-02,-8.494965732e-02,4.972323775e-02,6.251193583e-02,1.001451388e-01,-1.113965511e-01,8.469219506e-02,-4.730454832e-02};
//     Scalar epsilon_13=1.000000000e-03;
//     std::array<std::array<Scalar,10>,8> weights_14={{ {3.557049036e-01,-3.055827320e-01,1.127871871e-01,-2.397591472e-01,-2.704824507e-01,-4.088766575e-01,4.404658973e-01,5.896887183e-02,1.902821660e-02,4.324479401e-01},
//                                                        {1.936648935e-01,2.197610140e-01,-1.075418442e-01,4.595236182e-01,-2.878973782e-01,1.727001667e-01,-4.117991626e-01,1.439500898e-01,-4.489462674e-01,1.471630484e-01},
//                                                        {-8.817430586e-03,2.321404070e-01,-1.786624640e-02,2.158580273e-01,-1.186081097e-01,3.670343012e-02,2.060656548e-01,9.372919798e-02,-3.311883807e-01,-8.473055810e-02},
//                                                        {-1.413488686e-01,2.887615748e-02,-1.088175476e-01,-2.081313133e-01,-3.544175923e-01,-3.683956563e-01,3.934025764e-01,-4.640913010e-01,-2.598126046e-02,7.107881457e-02},
//                                                        {4.787643850e-01,3.398770690e-01,9.865088016e-02,2.096676379e-01,-3.177911341e-01,-2.357548028e-01,6.548528671e-01,-1.658320278e-01,5.137109160e-01,1.292318851e-01},
//                                                        {-5.350030959e-02,-4.461624920e-01,-6.401889771e-02,-3.313480914e-01,-1.844460666e-01,-2.648938894e-01,-1.101936102e-01,-1.195896976e-02,-5.400663614e-01,3.166174591e-01},
//                                                        {2.566511333e-01,2.259401381e-01,-1.160984207e-02,5.719654635e-02,-3.696546257e-01,-3.032363653e-01,3.789575100e-01,-2.816839516e-01,2.476725578e-01,1.222629026e-01},
//                                                        {4.279587269e-01,-1.670562848e-02,-1.624518186e-01,3.194932640e-01,-5.197309852e-01,-2.582280338e-01,-2.154778689e-01,2.549899518e-01,-2.731124759e-01,-8.791279048e-03}}};
//     std::array<Scalar,10> biases_14={4.376241192e-02,1.267522275e-01,8.299290389e-02,1.029757783e-01,9.762343019e-02,1.250658184e-02,1.064577550e-01,1.144008860e-01,9.739450365e-02,1.170560196e-01};

//     std::array<Scalar,3> layer_1_output;applyActivationFunctions<Scalar,3>(layer_1_output,model_input,&relu<Scalar>,0.0);
//     std::array<Scalar,3> layer_2_output;applyActivationFunctions<Scalar,3>(layer_2_output,layer_1_output,&linear<Scalar>,0.0);
//     std::array<Scalar,8> layer_3_output;forwardPass<Scalar,8,3>(layer_3_output.data(),layer_2_output.data(),weights_3,biases_3,&linear<Scalar>,0.0);
//     std::array<Scalar,8> layer_4_output;forwardPass<Scalar,8,8>(layer_4_output.data(),layer_3_output.data(),weights_4,biases_4,&silu<Scalar>,0.0);
//     std::array<Scalar,8> layer_5_output;layerNormalization<Scalar,8>(layer_5_output.data(),layer_4_output.data(),gamma_5.data(),beta_5.data(),epsilon_5);
//     std::array<Scalar,8> layer_6_output;forwardPass<Scalar,8,8>(layer_6_output.data(),layer_5_output.data(),weights_6,biases_6,&tanhCustom<Scalar>,0.0);
//     std::array<Scalar,8> layer_7_output;batchNormalization<Scalar,8>(layer_7_output.data(),layer_6_output.data(),gamma_7.data(),beta_7.data(),mean_7.data(),variance_7.data(),epsilon_7);
//     std::array<Scalar,8> layer_8_output;forwardPass<Scalar,8,8>(layer_8_output.data(),layer_7_output.data(),weights_8,biases_8,&linear<Scalar>,0.0);
//     std::array<Scalar,8> layer_9_output;applyActivationFunctions<Scalar,8>(layer_9_output,layer_8_output,&sigmoid<Scalar>,0.0);
//     std::array<Scalar,8> layer_10_output;applyActivationFunctions<Scalar,8>(layer_10_output,layer_9_output,&linear<Scalar>,0.0);
//     std::array<Scalar,8> layer_11_output;applyActivationFunctions<Scalar,8>(layer_11_output,layer_10_output,&elu<Scalar>,1.0);
//     std::array<Scalar,8> layer_12_output;forwardPass<Scalar,8,8>(layer_12_output.data(),layer_11_output.data(),weights_12,biases_12,&linear<Scalar>,0.0);
//     std::array<Scalar,8> layer_13_output;layerNormalization<Scalar,8>(layer_13_output.data(),layer_12_output.data(),gamma_13.data(),beta_13.data(),epsilon_13);
//     std::array<Scalar,10> layer_14_output;forwardPass<Scalar,10,8>(layer_14_output.data(),layer_13_output.data(),weights_14,biases_14,&linear<Scalar>,0.0);
//     std::array<Scalar,10> model_output=layer_14_output;
//     return model_output;
// }
















































// #include <iostream>
// #include <array>
// #include <random>
// #include <cmath>
// #include <utility>
// #include <immintrin.h>

// template<typename Scalar>
// using activationFunction=void(*)(Scalar&,Scalar,Scalar);

// template<typename Scalar>
// constexpr void relu(Scalar&o,Scalar i,Scalar a=0.0)noexcept{
//     o = i > 0 ? i : 0 ;
// }

// template<typename Scalar>
// constexpr void linear(Scalar&o,Scalar i,Scalar a=0.0)noexcept{
//     o=i;
// }

// template<typename Scalar>constexpr void silu(Scalar&o,Scalar i,Scalar a=1.0)noexcept{
//     Scalar s=1/(1+std::exp(-i));o=i*s;
// }

// template<typename Scalar>constexpr void tanhCustom(Scalar&o,Scalar i,Scalar a=0.0)noexcept{
//     o=std::tanh(i);
// }

// template<typename Scalar>constexpr void sigmoid(Scalar&o,Scalar i,Scalar a=0.0)noexcept{
//     o=1/(1+std::exp(-i));
// }

// template<typename Scalar>constexpr void elu(Scalar&o,Scalar i,Scalar a)noexcept{
//     o=i>0?i:a*(std::exp(i)-1);
// }

// template<typename Scalar>constexpr void addBias(Scalar&o,Scalar b)noexcept{
//     o+=b;
// }

// template<typename Scalar>constexpr void dotProduct(Scalar&s,Scalar in,Scalar w)noexcept{
//     s+=in*w;
// }

// template<typename Scalar,int size,std::size_t...Is>
// inline Scalar computeSum(const Scalar* in,std::index_sequence<Is...>)noexcept{
//     return((in[Is]+...));
// }

// template<typename Scalar,int size,std::size_t...Is>
// inline Scalar computeSumSquares(const Scalar* in,Scalar m,std::index_sequence<Is...>)noexcept{
//     return(((in[Is]-m)*(in[Is]-m))+...);
// }

// template<typename Scalar,int size,std::size_t...Is>
// inline void layerNormalizationImpl(Scalar* o,const Scalar* in,const Scalar* g,const Scalar* b,Scalar M,Scalar V,Scalar e,std::index_sequence<Is...>)noexcept{
//     ((o[Is]=g[Is]*((in[Is]-M)/Scalar(_mm_cvtsd_f64(_mm_sqrt_sd(_mm_setzero_pd(),_mm_set_sd(V+e)))))+b[Is]),...);
// }

// template<typename Scalar,int size>
// inline void layerNormalization(Scalar* o,const Scalar* in,const Scalar* g,const Scalar* B,Scalar e)noexcept{
//     Scalar M=computeSum<Scalar,size>(in,std::make_index_sequence<size>{})/static_cast<Scalar>(size);
//     Scalar V=computeSumSquares<Scalar,size>(in,M,std::make_index_sequence<size>{})/static_cast<Scalar>(size);
//     layerNormalizationImpl<Scalar,size>(o,in,g,B,M,V,e,std::make_index_sequence<size>{});
// }

// template<typename Scalar,int size,std::size_t...Is>
// inline void batchNormalizationImpl(Scalar* o,const Scalar* in,const Scalar* G,const Scalar* B,const Scalar* M,const Scalar* V,Scalar e,std::index_sequence<Is...>)noexcept{
//     ((o[Is]=G[Is]*((in[Is]-M[Is])/Scalar(_mm_cvtsd_f64(_mm_sqrt_sd(_mm_setzero_pd(),_mm_set_sd(V[Is]+e)))))+B[Is]),...);
// }

// template<typename Scalar,int size>
// inline void batchNormalization(Scalar* o,const Scalar* in,const Scalar* G,const Scalar* B,const Scalar* M,const Scalar* V,Scalar e)noexcept{
//     batchNormalizationImpl<Scalar,size>(o,in,G,B,M,V,e,std::make_index_sequence<size>{});
// }

// template<typename Scalar,int output_size,int input_size,std::size_t...Is>
// inline void forwardPassImpl(Scalar* o,const Scalar* in,const std::array<std::array<Scalar,output_size>,input_size>& w,const Scalar* b,activationFunction<Scalar> f,Scalar a,std::index_sequence<Is...>)noexcept{
//     (([&]{__m256d acc=_mm256_setzero_pd();for(int j=0;j<input_size;j+=4){__m256d vin=_mm256_loadu_pd(in+j);__m256d vw=_mm256_loadu_pd(&w[j][Is]);acc=_mm256_fmadd_pd(vin,vw,acc);}double tmp[4];_mm256_storeu_pd(tmp,acc);Scalar S=(tmp[0]+tmp[1]+tmp[2]+tmp[3]);S+=b[Is];f(o[Is],S,a);}()),...);
// }

// template<typename Scalar,int output_size,int input_size>
// inline void forwardPass(Scalar* o,const Scalar* in,const std::array<std::array<Scalar,output_size>,input_size>& w,const std::array<Scalar,output_size>& b,activationFunction<Scalar> f,Scalar a)noexcept{
//     forwardPassImpl<Scalar,output_size,input_size>(o,in,w,b.data(),f,a,std::make_index_sequence<output_size>{});
// }

// template<typename Scalar,std::size_t Size,std::size_t...Is>
// inline void applyActivationFunctionsImpl(std::array<Scalar,Size>& o,const std::array<Scalar,Size>& in,activationFunction<Scalar> f,Scalar a,std::index_sequence<Is...>)noexcept{
//     (f(o[Is],in[Is],a),...);
// }

// template<typename Scalar,std::size_t Size>
// inline void applyActivationFunctions(std::array<Scalar,Size>& o,const std::array<Scalar,Size>& in,activationFunction<Scalar> f,Scalar a)noexcept{
//     applyActivationFunctionsImpl<Scalar,Size>(o,in,f,a,std::make_index_sequence<Size>{});
// }

// template<typename Scalar=double>
// auto newArray(const std::array<Scalar,3>& initial_input){

//     std::array<Scalar,3> model_input=initial_input;

//     std::array<std::array<Scalar,8>,3> weights_3={{ 
//         {1.326323599e-01,-4.964405894e-01,-4.476362765e-01,4.825202525e-01,4.999661446e-01,6.104987264e-01,2.875608206e-01,-3.064861000e-01},
//         {3.347294331e-01,2.181584835e-01,-3.532409966e-01,6.674022079e-01,-6.745031476e-01,3.237143457e-01,-2.341475338e-01,-5.993123055e-01},
//         {1.393576264e-01,3.206691891e-02,-4.254285693e-01,6.793647408e-01,-6.585419774e-01,-6.157318950e-01,1.563780308e-01,-2.918911166e-02}}};

//     std::array<Scalar,8> biases_3={-9.336775169e-03,1.460986445e-03,1.543075219e-02,2.364202775e-02,-4.833743349e-02,4.356297478e-02,3.732505068e-02,-4.034891352e-02};

//     std::array<std::array<Scalar,8>,8> weights_4={{ 
//         {2.895489633e-01,1.404272467e-01,-5.511720181e-01,-1.963775754e-01,-4.318673015e-01,-6.175366044e-01,3.609300554e-01,-4.942331910e-01},
//         {1.385588199e-01,5.399128795e-01,-4.022693336e-01,4.465122521e-01,7.421918213e-02,4.528645426e-02,-2.302111685e-01,3.943967819e-01},
//         {-3.330304623e-01,9.097249061e-02,-3.297064081e-02,6.087180972e-01,-2.183382064e-01,-8.720761538e-02,5.392811298e-01,-3.553541601e-01},
//         {-4.102258384e-01,-3.817836344e-01,-3.408961594e-01,-5.070082545e-01,-9.703770280e-02,1.385468096e-01,-3.676263988e-01,1.009406447e-01},
//         {-7.793208212e-02,-2.422472984e-01,2.013003081e-01,4.458051324e-01,-6.043104529e-01,-1.048356891e-01,-1.337849349e-01,3.964666724e-01},
//         {-5.302410126e-01,-1.840705723e-01,-4.131732583e-01,4.750281870e-01,5.418121815e-01,-4.915194511e-01,2.186830342e-01,-5.351045132e-01},
//         {-5.902838111e-01,4.853354022e-02,-2.235947996e-01,3.651028574e-01,8.357726038e-02,-8.591035753e-02,1.432061195e-01,-2.278156132e-01},
//         {5.197193027e-01,-3.674283326e-01,4.082169831e-01,-1.086276844e-01,1.694692671e-01,1.661305875e-01,-1.906166077e-01,-4.951056540e-01}}};

//     std::array<Scalar,8> biases_4={-4.018265381e-02,7.817581296e-03,-2.712406032e-02,-2.441039495e-02,4.375640303e-02,2.640516497e-03,4.310625419e-02,2.533612400e-02};

//     std::array<Scalar,8> gamma_5={1.007159472e+00,9.102494121e-01,9.754046202e-01,1.009639382e+00,9.939744473e-01,1.053939223e+00,1.005302906e+00,1.049999237e+00};

//     std::array<Scalar,8> beta_5={-6.327476352e-02,1.301053464e-01,-8.200552315e-02,3.901715577e-02,1.436686069e-01,1.086948216e-01,9.198349435e-04,1.006997824e-01};

//     Scalar epsilon_5=1.000000000e-03;

//     std::array<std::array<Scalar,8>,8> weights_6={{ 
//         {-4.491657019e-01,1.051441580e-02,-8.537553996e-02,-4.117136896e-01,5.043860078e-01,6.056678295e-01,1.615756936e-02,-1.299536973e-01},
//         {3.373832703e-01,2.744476497e-01,2.461240739e-01,5.393412113e-01,-2.386006713e-01,-2.250907719e-01,4.689204097e-01,2.077780962e-01},
//         {-1.299372762e-01,-3.652926087e-01,1.432644129e-01,2.902889252e-01,-3.890721798e-01,-4.642286897e-01,-5.835254192e-01,-1.329570264e-01},
//         {5.750157312e-02,2.351277173e-01,8.795207739e-02,4.416449368e-01,4.325707555e-01,-4.251926541e-01,4.364499748e-01,-6.390288472e-01},
//         {5.611011386e-01,3.921681643e-01,3.791545928e-01,-2.027980834e-01,-1.832287610e-01,5.065606236e-01,5.255874395e-01,5.944129229e-01},
//         {4.012452364e-01,4.532038048e-02,-2.581494451e-01,-4.223471582e-01,-2.514020205e-01,4.273546338e-01,5.800563097e-01,4.546979666e-01},
//         {2.222973853e-01,1.506377310e-01,-9.531598538e-03,1.103633493e-01,-4.755297899e-01,4.269180000e-01,-4.701362550e-01,-5.456539392e-01},
//         {6.130970716e-01,1.676640958e-01,4.789482951e-01,-2.503974438e-01,-5.631949306e-01,3.413064778e-01,-4.982459545e-01,2.986344397e-01}}};

//     std::array<Scalar,8> biases_6={1.817649454e-01,1.338390857e-01,9.126558900e-02,2.131122723e-02,2.393466048e-02,-1.300014835e-02,3.099955246e-02,6.701175869e-02};

//     std::array<Scalar,8> gamma_7={8.193984032e-01,7.379534245e-01,9.682070017e-01,9.788198471e-01,9.873768091e-01,1.000282049e+00,9.579557776e-01,7.910596132e-01};

//     std::array<Scalar,8> beta_7={2.016024888e-01,-3.292292058e-01,3.209293187e-01,-3.260774910e-01,-1.682267338e-01,-1.893343627e-01,4.644928277e-01,-3.859341145e-01};

//     std::array<Scalar,8> mean_7={9.823068976e-01,8.114451170e-01,7.645511031e-01,-6.787924170e-01,-8.765227199e-01,9.079492688e-01,6.484096050e-01,9.349939227e-01};

//     std::array<Scalar,8> variance_7={1.048183767e-04,9.622686543e-03,1.916389167e-02,8.062008768e-02,2.308205934e-03,1.166001894e-02,6.997291744e-02,2.667167410e-02};

//     Scalar epsilon_7=1.000000000e-03;

//     std::array<std::array<Scalar,8>,8> weights_8={{ 
//         {-3.972038925e-01,-3.957073092e-01,-4.971534509e-05,-2.119336426e-01,-2.219672799e-01,3.227109313e-01,4.034543037e-01,3.877579272e-01},
//         {-2.669981420e-01,6.422920227e-01,-4.499044120e-01,1.564588845e-01,4.371322393e-01,4.035770893e-01,-2.477465421e-01,5.434113741e-01},
//         {6.102508307e-01,-6.751357317e-01,6.106441021e-01,-5.613425374e-01,2.262963206e-01,-2.247276157e-01,-8.041384816e-02,-1.671904474e-01},
//         {-4.067449868e-01,4.385202229e-01,1.891514361e-01,5.547617078e-01,2.592392862e-01,1.067967806e-02,2.351896018e-01,6.473272443e-01},
//         {-3.566615880e-01,-1.914600283e-01,-4.918406308e-01,1.386630833e-01,2.280944884e-01,5.265087485e-01,4.492689371e-01,-1.003818884e-01},
//         {-5.159434080e-01,-4.150543362e-02,2.560436428e-01,-4.054009914e-01,-2.460166346e-03,-5.730556846e-01,-7.491554320e-02,3.026012778e-01},
//         {-1.668728143e-01,-7.224540114e-01,5.997641012e-02,-2.716617882e-01,-1.031019539e-02,9.698459506e-02,-1.572506502e-02,-6.230728030e-01},
//         {-5.739236474e-01,5.079915524e-01,-4.854255319e-01,-7.777107507e-02,2.532823980e-01,3.523012996e-01,1.030851528e-01,3.888264596e-01}}};

//     std::array<Scalar,8> biases_8={2.145548761e-01,-4.947406054e-01,2.585793734e-01,-1.664437503e-01,-2.016923428e-01,2.338735759e-01,-1.787156314e-01,-1.901274323e-01};

//     std::array<std::array<Scalar,8>,8> weights_12={{ 
//         {-2.953823507e-01,9.143621475e-02,2.237670869e-01,-8.442353606e-01,6.365125179e-01,3.693207204e-01,-2.576244175e-01,-4.989452660e-01},
//         {4.503370821e-01,5.406127125e-02,5.430755764e-02,4.114288390e-01,-2.794103026e-01,-5.320529938e-01,-1.550755501e-01,3.113644421e-01},
//         {2.033043206e-01,4.824339449e-01,1.725191772e-01,-1.260260493e-01,7.400895357e-01,2.046696246e-01,-5.162982345e-01,-2.446578145e-01},
//         {2.942177653e-01,3.172022402e-01,-1.622629352e-02,-2.305077612e-01,-1.179132536e-01,-7.849524170e-02,3.792388737e-01,-2.122235298e-01},
//         {6.421005726e-01,4.533714652e-01,1.615806818e-01,1.940571330e-02,-7.184942812e-02,-2.896823883e-01,-1.747096479e-01,-2.893691361e-01},
//         {-1.758630760e-02,3.972255588e-01,3.065083176e-02,-7.126296759e-01,5.962981582e-01,1.485264003e-01,-4.742491543e-01,-6.893997788e-01},
//         {3.206037879e-01,2.579070926e-01,2.029581964e-01,-3.493468761e-01,5.849143490e-02,-1.007192209e-01,-4.437026102e-03,-4.000317752e-01},
//         {-5.845316499e-02,1.884575784e-01,-6.055310741e-02,-3.555367887e-01,1.380394958e-02,5.944981799e-02,2.185856849e-01,-2.417995185e-01}}};

//     std::array<Scalar,8> biases_12={1.869939864e-01,1.105956733e-01,1.053466052e-01,-3.331353962e-01,1.100314036e-01,-6.662753224e-02,-9.369239211e-03,-2.431712747e-01};

//     std::array<Scalar,8> gamma_13={7.639720440e-01,8.527194262e-01,6.928566098e-01,8.144926429e-01,9.072342515e-01,9.955703616e-01,8.884121776e-01,8.234179020e-01};

//     std::array<Scalar,8> beta_13={2.525599673e-02,-8.494965732e-02,4.972323775e-02,6.251193583e-02,1.001451388e-01,-1.113965511e-01,8.469219506e-02,-4.730454832e-02};

//     Scalar epsilon_13=1.000000000e-03;
    
//     std::array<std::array<Scalar,10>,8> weights_14={{ 
//         {3.557049036e-01,-3.055827320e-01,1.127871871e-01,-2.397591472e-01,-2.704824507e-01,-4.088766575e-01,4.404658973e-01,5.896887183e-02,1.902821660e-02,4.324479401e-01},
//         {1.936648935e-01,2.197610140e-01,-1.075418442e-01,4.595236182e-01,-2.878973782e-01,1.727001667e-01,-4.117991626e-01,1.439500898e-01,-4.489462674e-01,1.471630484e-01},
//         {-8.817430586e-03,2.321404070e-01,-1.786624640e-02,2.158580273e-01,-1.186081097e-01,3.670343012e-02,2.060656548e-01,9.372919798e-02,-3.311883807e-01,-8.473055810e-02},
//         {-1.413488686e-01,2.887615748e-02,-1.088175476e-01,-2.081313133e-01,-3.544175923e-01,-3.683956563e-01,3.934025764e-01,-4.640913010e-01,-2.598126046e-02,7.107881457e-02},
//         {4.787643850e-01,3.398770690e-01,9.865088016e-02,2.096676379e-01,-3.177911341e-01,-2.357548028e-01,6.548528671e-01,-1.658320278e-01,5.137109160e-01,1.292318851e-01},
//         {-5.350030959e-02,-4.461624920e-01,-6.401889771e-02,-3.313480914e-01,-1.844460666e-01,-2.648938894e-01,-1.101936102e-01,-1.195896976e-02,-5.400663614e-01,3.166174591e-01},
//         {2.566511333e-01,2.259401381e-01,-1.160984207e-02,5.719654635e-02,-3.696546257e-01,-3.032363653e-01,3.789575100e-01,-2.816839516e-01,2.476725578e-01,1.222629026e-01},
//         {4.279587269e-01,-1.670562848e-02,-1.624518186e-01,3.194932640e-01,-5.197309852e-01,-2.582280338e-01,-2.154778689e-01,2.549899518e-01,-2.731124759e-01,-8.791279048e-03}}};

//     std::array<Scalar,10> biases_14={4.376241192e-02,1.267522275e-01,8.299290389e-02,1.029757783e-01,9.762343019e-02,1.250658184e-02,1.064577550e-01,1.144008860e-01,9.739450365e-02,1.170560196e-01};

//     std::array<Scalar,3> layer_1_output;
//     applyActivationFunctions<Scalar,3>(layer_1_output,model_input,&relu<Scalar>,0.0);

//     std::array<Scalar,3> layer_2_output;
//     applyActivationFunctions<Scalar,3>(layer_2_output,layer_1_output,&linear<Scalar>,0.0);

//     std::array<Scalar,8> layer_3_output;
//     forwardPass<Scalar,8,3>(layer_3_output.data(),layer_2_output.data(),weights_3,biases_3,&linear<Scalar>,0.0);

//     std::array<Scalar,8> layer_4_output;
//     forwardPass<Scalar,8,8>(layer_4_output.data(),layer_3_output.data(),weights_4,biases_4,&silu<Scalar>,0.0);

//     std::array<Scalar,8> layer_5_output;
//     layerNormalization<Scalar,8>(layer_5_output.data(),layer_4_output.data(),gamma_5.data(),beta_5.data(),epsilon_5);

//     std::array<Scalar,8> layer_6_output;
//     forwardPass<Scalar,8,8>(layer_6_output.data(),layer_5_output.data(),weights_6,biases_6,&tanhCustom<Scalar>,0.0);

//     std::array<Scalar,8> layer_7_output;
//     batchNormalization<Scalar,8>(layer_7_output.data(),layer_6_output.data(),gamma_7.data(),beta_7.data(),mean_7.data(),variance_7.data(),epsilon_7);

//     std::array<Scalar,8> layer_8_output;
//     forwardPass<Scalar,8,8>(layer_8_output.data(),layer_7_output.data(),weights_8,biases_8,&linear<Scalar>,0.0);

//     std::array<Scalar,8> layer_9_output;
//     applyActivationFunctions<Scalar,8>(layer_9_output,layer_8_output,&sigmoid<Scalar>,0.0);

//     std::array<Scalar,8> layer_10_output;
//     applyActivationFunctions<Scalar,8>(layer_10_output,layer_9_output,&linear<Scalar>,0.0);

//     std::array<Scalar,8> layer_11_output;
//     applyActivationFunctions<Scalar,8>(layer_11_output,layer_10_output,&elu<Scalar>,1.0);

//     std::array<Scalar,8> layer_12_output;
//     forwardPass<Scalar,8,8>(layer_12_output.data(),layer_11_output.data(),weights_12,biases_12,&linear<Scalar>,0.0);

//     std::array<Scalar,8> layer_13_output;
//     layerNormalization<Scalar,8>(layer_13_output.data(),layer_12_output.data(),gamma_13.data(),beta_13.data(),epsilon_13);

//     std::array<Scalar,10> layer_14_output;
//     forwardPass<Scalar,10,8>(layer_14_output.data(),layer_13_output.data(),weights_14,biases_14,&linear<Scalar>,0.0);

//     std::array<Scalar,10> model_output=layer_14_output;

//     return model_output;
// }










































// newArray.h
#ifndef NEWARRAY_H
#define NEWARRAY_H

#include <iostream>
#include <array>
#include <cmath>
#include <utility>

// Define an activation function type
template<typename Scalar>
using activationFunction = void(*)(Scalar&, Scalar, Scalar);

// Activation Functions
template<typename Scalar>
constexpr void relu(Scalar& o, Scalar i, Scalar a = 0.0) noexcept {
    o = (i > 0) ? i : 0;
}

template<typename Scalar>
constexpr void linear(Scalar& o, Scalar i, Scalar a = 0.0) noexcept {
    o = i;
}

template<typename Scalar>
constexpr void silu(Scalar& o, Scalar i, Scalar a = 1.0) noexcept {
    Scalar s = 1 / (1 + std::exp(-i));
    o = i * s;
}

template<typename Scalar>
constexpr void tanhCustom(Scalar& o, Scalar i, Scalar a = 0.0) noexcept {
    o = std::tanh(i);
}

template<typename Scalar>
constexpr void sigmoid(Scalar& o, Scalar i, Scalar a = 0.0) noexcept {
    o = 1 / (1 + std::exp(-i));
}

template<typename Scalar>
constexpr void elu(Scalar& o, Scalar i, Scalar a) noexcept {
    o = (i > 0) ? i : a * (std::exp(i) - 1);
}

// Utility Functions
template<typename Scalar>
constexpr void addBias(Scalar& o, Scalar b) noexcept {
    o += b;
}

template<typename Scalar>
constexpr void dotProduct(Scalar& s, Scalar in, Scalar w) noexcept {
    s += in * w;
}

// Compute the sum of elements in an array
template<typename Scalar, int size, std::size_t...Is>
inline Scalar computeSum(const Scalar* in, std::index_sequence<Is...>) noexcept {
    return ((in[Is] + ...));
}

// Compute the sum of squares of elements in an array after subtracting the mean
template<typename Scalar, int size, std::size_t...Is>
inline Scalar computeSumSquares(const Scalar* in, Scalar m, std::index_sequence<Is...>) noexcept {
    return (((in[Is] - m) * (in[Is] - m)) + ...);
}

// Layer Normalization Implementation
template<typename Scalar, int size, std::size_t...Is>
inline void layerNormalizationImpl(Scalar* o, const Scalar* in, const Scalar* g, const Scalar* b, Scalar M, Scalar V, Scalar e, std::index_sequence<Is...>) noexcept {
    ((o[Is] = g[Is] * ((in[Is] - M) / std::sqrt(V + e)) + b[Is]), ...);
}

template<typename Scalar, int size>
inline void layerNormalization(Scalar* o, const Scalar* in, const Scalar* g, const Scalar* B, Scalar e) noexcept {
    Scalar M = computeSum<Scalar, size>(in, std::make_index_sequence<size>{}) / static_cast<Scalar>(size);
    Scalar V = computeSumSquares<Scalar, size>(in, M, std::make_index_sequence<size>{}) / static_cast<Scalar>(size);
    layerNormalizationImpl<Scalar, size>(o, in, g, B, M, V, e, std::make_index_sequence<size>{});
}

// Batch Normalization Implementation (per feature)
template<typename Scalar, int size, std::size_t...Is>
inline void batchNormalizationImpl(Scalar* o, const Scalar* in, const Scalar* G, const Scalar* B, const Scalar* M, const Scalar* V, Scalar e, std::index_sequence<Is...>) noexcept {
    ((o[Is] = G[Is] * ((in[Is] - M[Is]) / std::sqrt(V[Is] + e)) + B[Is]), ...);
}

template<typename Scalar, int size>
inline void batchNormalization(Scalar* o, const Scalar* in, const Scalar* G, const Scalar* B, const Scalar* M, const Scalar* V, Scalar e) noexcept {
    batchNormalizationImpl<Scalar, size>(o, in, G, B, M, V, e, std::make_index_sequence<size>{});
}

// Forward Pass Implementation with Compiler-Specific Pragmas
template<typename Scalar, int output_size, int input_size>
inline void forwardPass(
    Scalar* o,
    const Scalar* in,
    const std::array<std::array<Scalar, output_size>, input_size>& w,
    const std::array<Scalar, output_size>& b,
    activationFunction<Scalar> f,
    Scalar a
) noexcept {
    // Compiler-specific pragmas for vectorization
    #if defined(__clang__)
        #pragma clang loop vectorize(enable) interleave(enable) unroll(enable)
    #elif defined(__GNUC__) && !defined(__clang__)
        #pragma GCC ivdep
        #pragma GCC vectorize (always)
    #endif

    for (int i = 0; i < output_size; ++i) {
        Scalar S = 0;
        for(int j = 0; j < input_size; ++j) {
            S += in[j] * w[j][i];
        }
        S += b[i];
        f(o[i], S, a);
    }
}

// Apply Activation Functions Implementation
template<typename Scalar, std::size_t Size, std::size_t...Is>
inline void applyActivationFunctionsImpl(
    std::array<Scalar, Size>& o,
    const std::array<Scalar, Size>& in,
    activationFunction<Scalar> f,
    Scalar a,
    std::index_sequence<Is...>
) noexcept {
    (f(o[Is], in[Is], a), ...);
}

template<typename Scalar, std::size_t Size>
inline void applyActivationFunctions(
    std::array<Scalar, Size>& o,
    const std::array<Scalar, Size>& in,
    activationFunction<Scalar> f,
    Scalar a
) noexcept {
    applyActivationFunctionsImpl<Scalar, Size>(o, in, f, a, std::make_index_sequence<Size>{});
}

// The newArray Function: Constructs and Executes the Neural Network Model
template<typename Scalar = double>
std::array<Scalar, 10> newArray(const std::array<Scalar, 3>& initial_input) {
    // Initialize model input
    std::array<Scalar, 3> model_input = initial_input;

    // Define weights and biases for various layers

    // Layer 3
    std::array<std::array<Scalar, 8>, 3> weights_3 = {{
        {1.326323599e-01, -4.964405894e-01, -4.476362765e-01, 4.825202525e-01, 4.999661446e-01, 6.104987264e-01, 2.875608206e-01, -3.064861000e-01},
        {3.347294331e-01, 2.181584835e-01, -3.532409966e-01, 6.674022079e-01, -6.745031476e-01, 3.237143457e-01, -2.341475338e-01, -5.993123055e-01},
        {1.393576264e-01, 3.206691891e-02, -4.254285693e-01, 6.793647408e-01, -6.585419774e-01, -6.157318950e-01, 1.563780308e-01, -2.918911166e-02}
    }};
    std::array<Scalar, 8> biases_3 = {-9.336775169e-03, 1.460986445e-03, 1.543075219e-02, 2.364202775e-02, -4.833743349e-02, 4.356297478e-02, 3.732505068e-02, -4.034891352e-02};

    // Layer 4
    std::array<std::array<Scalar, 8>, 8> weights_4 = {{
        {2.895489633e-01, 1.404272467e-01, -5.511720181e-01, -1.963775754e-01, -4.318673015e-01, -6.175366044e-01, 3.609300554e-01, -4.942331910e-01},
        {1.385588199e-01, 5.399128795e-01, -4.022693336e-01, 4.465122521e-01, 7.421918213e-02, 4.528645426e-02, -2.302111685e-01, 3.943967819e-01},
        {-3.330304623e-01, 9.097249061e-02, -3.297064081e-02, 6.087180972e-01, -2.183382064e-01, -8.720761538e-02, 5.392811298e-01, -3.553541601e-01},
        {-4.102258384e-01, -3.817836344e-01, -3.408961594e-01, -5.070082545e-01, -9.703770280e-02, 1.385468096e-01, -3.676263988e-01, 1.009406447e-01},
        {-7.793208212e-02, -2.422472984e-01, 2.013003081e-01, 4.458051324e-01, -6.043104529e-01, -1.048356891e-01, -1.337849349e-01, 3.964666724e-01},
        {-5.302410126e-01, -1.840705723e-01, -4.131732583e-01, 4.750281870e-01, 5.418121815e-01, -4.915194511e-01, 2.186830342e-01, -5.351045132e-01},
        {-5.902838111e-01, 4.853354022e-02, -2.235947996e-01, 3.651028574e-01, 8.357726038e-02, -8.591035753e-02, 1.432061195e-01, -2.278156132e-01},
        {5.197193027e-01, -3.674283326e-01, 4.082169831e-01, -1.086276844e-01, 1.694692671e-01, 1.661305875e-01, -1.906166077e-01, -4.951056540e-01}
    }};
    std::array<Scalar, 8> biases_4 = {-4.018265381e-02, 7.817581296e-03, -2.712406032e-02, -2.441039495e-02, 4.375640303e-02, 2.640516497e-03, 4.310625419e-02, 2.533612400e-02};

    // Layer 5
    std::array<Scalar, 8> gamma_5 = {1.007159472e+00, 9.102494121e-01, 9.754046202e-01, 1.009639382e+00, 9.939744473e-01, 1.053939223e+00, 1.005302906e+00, 1.049999237e+00};
    std::array<Scalar, 8> beta_5 = {-6.327476352e-02, 1.301053464e-01, -8.200552315e-02, 3.901715577e-02, 1.436686069e-01, 1.086948216e-01, 9.198349435e-04, 1.006997824e-01};
    Scalar epsilon_5 = 1.000000000e-03;

    // Layer 6
    std::array<std::array<Scalar, 8>, 8> weights_6 = {{
        {-4.491657019e-01, 1.051441580e-02, -8.537553996e-02, -4.117136896e-01, 5.043860078e-01, 6.056678295e-01, 1.615756936e-02, -1.299536973e-01},
        {3.373832703e-01, 2.744476497e-01, 2.461240739e-01, 5.393412113e-01, -2.386006713e-01, -2.250907719e-01, 4.689204097e-01, 2.077780962e-01},
        {-1.299372762e-01, -3.652926087e-01, 1.432644129e-01, 2.902889252e-01, -3.890721798e-01, -4.642286897e-01, -5.835254192e-01, -1.329570264e-01},
        {5.750157312e-02, 2.351277173e-01, 8.795207739e-02, 4.416449368e-01, 4.325707555e-01, -4.251926541e-01, 4.364499748e-01, -6.390288472e-01},
        {5.611011386e-01, 3.921681643e-01, 3.791545928e-01, -2.027980834e-01, -1.832287610e-01, 5.065606236e-01, 5.255874395e-01, 5.944129229e-01},
        {4.012452364e-01, 4.532038048e-02, -2.581494451e-01, -4.223471582e-01, -2.514020205e-01, 4.273546338e-01, 5.800563097e-01, 4.546979666e-01},
        {2.222973853e-01, 1.506377310e-01, -9.531598538e-03, 1.103633493e-01, -4.755297899e-01, 4.269180000e-01, -4.701362550e-01, -5.456539392e-01},
        {6.130970716e-01, 1.676640958e-01, 4.789482951e-01, -2.503974438e-01, -5.631949306e-01, 3.413064778e-01, -4.982459545e-01, 2.986344397e-01}
    }};
    std::array<Scalar, 8> biases_6 = {1.817649454e-01, 1.338390857e-01, 9.126558900e-02, 2.131122723e-02, 2.393466048e-02, -1.300014835e-02, 3.099955246e-02, 6.701175869e-02};

    // Layer 7
    std::array<Scalar, 8> gamma_7 = {8.193984032e-01, 7.379534245e-01, 9.682070017e-01, 9.788198471e-01, 9.873768091e-01, 1.000282049e+00, 9.579557776e-01, 7.910596132e-01};
    std::array<Scalar, 8> beta_7 = {2.016024888e-01, -3.292292058e-01, 3.209293187e-01, -3.260774910e-01, -1.682267338e-01, -1.893343627e-01, 4.644928277e-01, -3.859341145e-01};
    std::array<Scalar, 8> mean_7 = {9.823068976e-01, 8.114451170e-01, 7.645511031e-01, -6.787924170e-01, -8.765227199e-01, 9.079492688e-01, 6.484096050e-01, 9.349939227e-01};
    std::array<Scalar, 8> variance_7 = {1.048183767e-04, 9.622686543e-03, 1.916389167e-02, 8.062008768e-02, 2.308205934e-03, 1.166001894e-02, 6.997291744e-02, 2.667167410e-02};
    Scalar epsilon_7 = 1.000000000e-03;

    // Layer 8
    std::array<std::array<Scalar, 8>, 8> weights_8 = {{
        {-3.972038925e-01, -3.957073092e-01, -4.971534509e-05, -2.119336426e-01, -2.219672799e-01, 3.227109313e-01, 4.034543037e-01, 3.877579272e-01},
        {-2.669981420e-01, 6.422920227e-01, -4.499044120e-01, 1.564588845e-01, 4.371322393e-01, 4.035770893e-01, -2.477465421e-01, 5.434113741e-01},
        {6.102508307e-01, -6.751357317e-01, 6.106441021e-01, -5.613425374e-01, 2.262963206e-01, -2.247276157e-01, -8.041384816e-02, -1.671904474e-01},
        {-4.067449868e-01, 4.385202229e-01, 1.891514361e-01, 5.547617078e-01, 2.592392862e-01, 1.067967806e-02, 2.351896018e-01, 6.473272443e-01},
        {-3.566615880e-01, -1.914600283e-01, -4.918406308e-01, 1.386630833e-01, 2.280944884e-01, 5.265087485e-01, 4.492689371e-01, -1.003818884e-01},
        {-5.159434080e-01, -4.150543362e-02, 2.560436428e-01, -4.054009914e-01, -2.460166346e-03, -5.730556846e-01, -7.491554320e-02, 3.026012778e-01},
        {-1.668728143e-01, -7.224540114e-01, 5.997641012e-02, -2.716617882e-01, -1.031019539e-02, 9.698459506e-02, -1.572506502e-02, -6.230728030e-01},
        {-5.739236474e-01, 5.079915524e-01, -4.854255319e-01, -7.777107507e-02, 2.532823980e-01, 3.523012996e-01, 1.030851528e-01, 3.888264596e-01}
    }};
    std::array<Scalar, 8> biases_8 = {2.145548761e-01, -4.947406054e-01, 2.585793734e-01, -1.664437503e-01, -2.016923428e-01, 2.338735759e-01, -1.787156314e-01, -1.901274323e-01};

    // Layer 12
    std::array<std::array<Scalar, 8>, 8> weights_12 = {{
        {-2.953823507e-01, 9.143621475e-02, 2.237670869e-01, -8.442353606e-01, 6.365125179e-01, 3.693207204e-01, -2.576244175e-01, -4.989452660e-01},
        {4.503370821e-01, 5.406127125e-02, 5.430755764e-02, 4.114288390e-01, -2.794103026e-01, -5.320529938e-01, -1.550755501e-01, 3.113644421e-01},
        {2.033043206e-01, 4.824339449e-01, 1.725191772e-01, -1.260260493e-01, 7.400895357e-01, 2.046696246e-01, -5.162982345e-01, -2.446578145e-01},
        {2.942177653e-01, 3.172022402e-01, -1.622629352e-02, -2.305077612e-01, -1.179132536e-01, -7.849524170e-02, 3.792388737e-01, -2.122235298e-01},
        {6.421005726e-01, 4.533714652e-01, 1.615806818e-01, 1.940571330e-02, -7.184942812e-02, -2.896823883e-01, -1.747096479e-01, -2.893691361e-01},
        {-1.758630760e-02, 3.972255588e-01, 3.065083176e-02, -7.126296759e-01, 5.962981582e-01, 1.485264003e-01, -4.742491543e-01, -6.893997788e-01},
        {3.206037879e-01, 2.579070926e-01, 2.029581964e-01, -3.493468761e-01, 5.849143490e-02, -1.007192209e-01, -4.437026102e-03, -4.000317752e-01},
        {-5.845316499e-02, 1.884575784e-01, -6.055310741e-02, -3.555367887e-01, 1.380394958e-02, 5.944981799e-02, 2.185856849e-01, -2.417995185e-01}
    }};
    std::array<Scalar, 8> biases_12 = {1.869939864e-01, 1.105956733e-01, 1.053466052e-01, -3.331353962e-01, 1.100314036e-01, -6.662753224e-02, -9.369239211e-03, -2.431712747e-01};

    // Layer 14
    std::array<std::array<Scalar, 10>, 8> weights_14 = {{
        {3.557049036e-01, -3.055827320e-01, 1.127871871e-01, -2.397591472e-01, -2.704824507e-01, -4.088766575e-01, 4.404658973e-01, 5.896887183e-02, 1.902821660e-02, 4.324479401e-01},
        {1.936648935e-01, 2.197610140e-01, -1.075418442e-01, 4.595236182e-01, -2.878973782e-01, 1.727001667e-01, -4.117991626e-01, 1.439500898e-01, -4.489462674e-01, 1.471630484e-01},
        {-8.817430586e-03, 2.321404070e-01, -1.786624640e-02, 2.158580273e-01, -1.186081097e-01, 3.670343012e-02, 2.060656548e-01, 9.372919798e-02, -3.311883807e-01, -8.473055810e-02},
        {-1.413488686e-01, 2.887615748e-02, -1.088175476e-01, -2.081313133e-01, -3.544175923e-01, -3.683956563e-01, 3.934025764e-01, -4.640913010e-01, -2.598126046e-02, 7.107881457e-02},
        {4.787643850e-01, 3.398770690e-01, 9.865088016e-02, 2.096676379e-01, -3.177911341e-01, -2.357548028e-01, 6.548528671e-01, -1.658320278e-01, 5.137109160e-01, 1.292318851e-01},
        {-5.350030959e-02, -4.461624920e-01, -6.401889771e-02, -3.313480914e-01, -1.844460666e-01, -2.648938894e-01, -1.101936102e-01, -1.195896976e-02, -5.400663614e-01, 3.166174591e-01},
        {2.566511333e-01, 2.259401381e-01, -1.160984207e-02, 5.719654635e-02, -3.696546257e-01, -3.032363653e-01, 3.789575100e-01, -2.816839516e-01, 2.476725578e-01, 1.222629026e-01},
        {4.279587269e-01, -1.670562848e-02, -1.624518186e-01, 3.194932640e-01, -5.197309852e-01, -2.582280338e-01, -2.154778689e-01, 2.549899518e-01, -2.731124759e-01, -8.791279048e-03}
    }};
    std::array<Scalar, 10> biases_14 = {4.376241192e-02, 1.267522275e-01, 8.299290389e-02, 1.029757783e-01, 9.762343019e-02, 1.250658184e-02, 1.064577550e-01, 1.144008860e-01, 9.739450365e-02, 1.170560196e-01};

    // Layer 13
    std::array<Scalar, 8> gamma_13 = {7.639720440e-01, 8.527194262e-01, 6.928566098e-01, 8.144926429e-01, 9.072342515e-01, 9.955703616e-01, 8.884121776e-01, 8.234179020e-01};
    std::array<Scalar, 8> beta_13 = {2.525599673e-02, -8.494965732e-02, 4.972323775e-02, 6.251193583e-02, 1.001451388e-01, -1.113965511e-01, 8.469219506e-02, -4.730454832e-02};
    Scalar epsilon_13 = 1.000000000e-03;

    // Layers Execution

    // Layer 1: Apply ReLU Activation
    std::array<Scalar, 3> layer_1_output;
    applyActivationFunctions<Scalar, 3>(layer_1_output, model_input, &relu<Scalar>, 0.0);

    // Layer 2: Apply Linear Activation
    std::array<Scalar, 3> layer_2_output;
    applyActivationFunctions<Scalar, 3>(layer_2_output, layer_1_output, &linear<Scalar>, 0.0);

    // Layer 3: Forward Pass (Layer 3)
    std::array<Scalar, 8> layer_3_output;
    forwardPass<Scalar, 8, 3>(layer_3_output.data(), layer_2_output.data(), weights_3, biases_3, &linear<Scalar>, 0.0);

    // Layer 4: Forward Pass (Layer 4) with SiLU Activation
    std::array<Scalar, 8> layer_4_output;
    forwardPass<Scalar, 8, 8>(layer_4_output.data(), layer_3_output.data(), weights_4, biases_4, &silu<Scalar>, 0.0);

    // Layer 5: Layer Normalization
    std::array<Scalar, 8> layer_5_output;
    layerNormalization<Scalar, 8>(layer_5_output.data(), layer_4_output.data(), gamma_5.data(), beta_5.data(), epsilon_5);

    // Layer 6: Forward Pass (Layer 6) with Tanh Activation
    std::array<Scalar, 8> layer_6_output;
    forwardPass<Scalar, 8, 8>(layer_6_output.data(), layer_5_output.data(), weights_6, biases_6, &tanhCustom<Scalar>, 0.0);

    // Layer 7: Batch Normalization
    std::array<Scalar, 8> layer_7_output;
    batchNormalization<Scalar, 8>(layer_7_output.data(), layer_6_output.data(), gamma_7.data(), beta_7.data(), mean_7.data(), variance_7.data(), epsilon_7);

    // Layer 8: Forward Pass (Layer 8)
    std::array<Scalar, 8> layer_8_output;
    forwardPass<Scalar, 8, 8>(layer_8_output.data(), layer_7_output.data(), weights_8, biases_8, &linear<Scalar>, 0.0);

    // Layer 9: Apply Sigmoid Activation
    std::array<Scalar, 8> layer_9_output;
    applyActivationFunctions<Scalar, 8>(layer_9_output, layer_8_output, &sigmoid<Scalar>, 0.0);

    // Layer 10: Apply Linear Activation
    std::array<Scalar, 8> layer_10_output;
    applyActivationFunctions<Scalar, 8>(layer_10_output, layer_9_output, &linear<Scalar>, 0.0);

    // Layer 11: Apply ELU Activation with alpha = 1.0
    std::array<Scalar, 8> layer_11_output;
    applyActivationFunctions<Scalar, 8>(layer_11_output, layer_10_output, &elu<Scalar>, 1.0);

    // Layer 12: Forward Pass (Layer 12)
    std::array<Scalar, 8> layer_12_output;
    forwardPass<Scalar, 8, 8>(layer_12_output.data(), layer_11_output.data(), weights_12, biases_12, &linear<Scalar>, 0.0);

    // Layer 13: Layer Normalization
    std::array<Scalar, 8> layer_13_output;
    layerNormalization<Scalar, 8>(layer_13_output.data(), layer_12_output.data(), gamma_13.data(), beta_13.data(), epsilon_13);

    // Layer 14: Forward Pass (Layer 14)
    std::array<Scalar, 10> layer_14_output;
    forwardPass<Scalar, 10, 8>(layer_14_output.data(), layer_13_output.data(), weights_14, biases_14, &linear<Scalar>, 0.0);

    // Final Model Output
    std::array<Scalar, 10> model_output = layer_14_output;

    return model_output;
}

#endif // NEWARRAY_H