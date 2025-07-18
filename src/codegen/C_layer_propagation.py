import os
import re
import absl.logging
import warnings

absl.logging.set_verbosity("error")
warnings.filterwarnings("ignore", category=UserWarning, module="keras")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def layer_propagation(cpp_code, activation_functions, layer_type, base_file_name):
    """
    Generate C++ lambda-based activation functions (with no indentation for the lambdas)
    and normalization functions. ForwardPass also remains as Code 2 style.
    """

    # regular forward pass
    dense_function = {
        "Dense": f"""
template<typename Scalar, int output_size, typename ActFun>
inline void Dense_{base_file_name}(Scalar* __restrict outputs, const Scalar* __restrict inputs, const Scalar * __restrict weights, const Scalar * __restrict biases, int input_size, ActFun activation_function, Scalar alpha) noexcept 
{{
    for(int i = 0; i < output_size; ++i){{
        Scalar sum = 0;
        
        for(int j = 0; j < input_size; ++j){{
            sum += inputs[j] * weights[j * output_size + i];
        }}
        sum += biases[i];
        activation_function(outputs[i], sum, alpha);
    }}
}}
"""
    }

    # reshape functions
    reshape_functions = {
        "Reshape": f"""
template<typename Scalar, int N>
inline void Reshape_{base_file_name}(Scalar * __restrict outputs, const Scalar * __restrict inputs) noexcept 
{{
    for (int i = 0; i < N; ++i) {{
        outputs[i] = inputs[i];
    }}
}}
"""
    }

    # preprocessing functions
    preprocessing_functions = {
        "Rescale": f"""
template<typename Scalar, int output_size>
inline void Rescale_{base_file_name}(Scalar * __restrict outputs, const Scalar * __restrict inputs, const Scalar * __restrict scale, const Scalar * __restrict offset) noexcept 
{{
    for (int i = 0; i < output_size; ++i) {{
        outputs[i] = inputs[i] * scale[i] + offset[i];
    }}
}}
"""
    }

    # lambda activation functions
    lambda_functions = {
        "relu": """
    auto relu = +[](Scalar& output, Scalar input, Scalar alpha) noexcept 
    {
        output = input > 0 ? input : 0;
    };
""",
        "sigmoid": """
    auto sigmoid = +[](Scalar& output, Scalar input, Scalar alpha) noexcept 
    {
        output = 1 / (1 + std::exp(-input));
    };
""",
        "tanhCustom": """
    auto tanhCustom = +[](Scalar& output, Scalar input, Scalar alpha) noexcept 
    {
        output = std::tanh(input);
    };
""",
        "leakyrelu": """
    auto leakyrelu = +[](Scalar& output, Scalar input, Scalar alpha) noexcept 
    {
        output = input > 0 ? input : alpha * input;
    };
""",
        "linear": """
    auto linear = +[](Scalar& output, Scalar input, Scalar alpha) noexcept 
    {
        output = input;
    };
""",
        "elu": """
    auto elu = +[](Scalar& output, Scalar input, Scalar alpha) noexcept 
    {
        output = input > 0 ? input : alpha * (std::exp(input) - 1);
    };
""",
        "selu": """
    auto selu = +[](Scalar& output, Scalar input, Scalar alpha) noexcept 
    {
        output = 1.0507009873554804934193349852946 * (input > 0 ? input : 1.6732632423543772848170429916717 * (std::exp(input) - 1));
    };
""",
        "swish": """
    auto swish = +[](Scalar& output, Scalar input, Scalar alpha) noexcept 
    {
        output = input / (1 + std::exp(-alpha * input));
    };
""",
        "prelu": """
    auto prelu = +[](Scalar& output, Scalar input, Scalar alpha) noexcept 
    {
        output = input > 0 ? input : alpha * input;
    };
""",
        "silu": """
    auto silu = +[](Scalar& output, Scalar input, Scalar alpha) noexcept 
    {
        auto sigmoid = 1 / (1 + std::exp(-input));
        output = input * sigmoid;
    };
""",
        "gelu": """
    static constexpr Scalar kC0 = 0.044715;
    static constexpr Scalar kSqrt2PiInv = Scalar(0.7978845608028654);
    auto gelu = +[](Scalar& output, Scalar input, Scalar alpha) noexcept 
    {
        Scalar x3 = input * input * input;
        Scalar y  = kSqrt2PiInv * (input + kC0 * x3);
        output     = Scalar(0.5) * input * (Scalar(1) + std::tanh(y));
    };
""",
        "softmax": """
    auto softmax = +[](Scalar * __restrict outputs, Scalar * __restrict inputs, int size) noexcept 
    {
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
""",
        "mish": """
    auto mish = +[](Scalar& output, Scalar input, Scalar alpha) noexcept 
    {
        Scalar softplus;
        if (input > Scalar(20))          
            softplus = input;
        else if (input < Scalar(-20))    
            softplus = std::exp(input);
        else
            softplus = std::log1p(std::exp(input));
        output = input * std::tanh(softplus);
    };
""",
        "softplus": """
    auto softplus = +[](Scalar& output, Scalar input, Scalar alpha) noexcept
    {
        // output = std::log1p(std::exp(input));
        output = (input > 20) ? input : std::log1p(std::exp(input));
    };
"""
    }

    # normalization functions
    normalization_functions = {
        "LayerNormalization": f"""
template <typename Scalar, int size>
inline void LayerNormalization_{base_file_name}(Scalar * __restrict outputs, const Scalar * __restrict inputs, const Scalar * __restrict gamma, const Scalar * __restrict beta, Scalar epsilon) noexcept
{{
    Scalar mean = 0;
    Scalar variance = 0;
    
    for (int i = 0; i < size; ++i)
    {{
        mean += inputs[i];
    }}
    mean /= size;
    
    for (int i = 0; i < size; ++i)
    {{
        variance += (inputs[i] - mean) * (inputs[i] - mean);
    }}
    variance /= size;
    
    for (int i = 0; i < size; ++i)
    {{
        outputs[i] = gamma[i] * ((inputs[i] - mean) / std::sqrt(variance + epsilon)) + beta[i];
    }}
}}
""",
        "BatchNormalization": f"""
template <typename Scalar, int size>
inline void BatchNormalization_{base_file_name}(Scalar * __restrict outputs, const Scalar * __restrict inputs, const Scalar * __restrict gamma, const Scalar * __restrict beta, const Scalar * __restrict mean, const Scalar * __restrict variance, const Scalar epsilon) noexcept
{{
    
    for (int i = 0; i < size; ++i)
    {{
        outputs[i] = gamma[i] * ((inputs[i] - mean[i]) / std::sqrt(variance[i] + epsilon)) + beta[i];
    }}
}}
""",
        "BatchNormalization2D": f"""
template <typename Scalar, int channels, int height, int width>
inline void BatchNormalization2D_{base_file_name}(Scalar * __restrict outputs, const Scalar * __restrict inputs,
                          const Scalar * __restrict gamma, const Scalar * __restrict beta,
                          const Scalar * __restrict mean, const Scalar * __restrict variance,
                          Scalar epsilon) noexcept
{{
    for (int c = 0; c < channels; ++c)
    {{
        
        for (int i = 0; i < height * width; ++i)
        {{
            int idx = i * channels + c;
            outputs[idx] = gamma[c] * ((inputs[idx] - mean[c]) / std::sqrt(variance[c] + epsilon)) +
                           beta[c];
        }}
    }}
}}
""",
        "LayerNormalization2D": f"""
template <typename Scalar, int channels, int height, int width>
inline void LayerNormalization2D_{base_file_name}(Scalar * __restrict outputs, const Scalar * __restrict inputs,
                          const Scalar * __restrict gamma, const Scalar * __restrict beta,
                          Scalar epsilon) noexcept
{{
    for (int c = 0; c < channels; ++c)
    {{
        Scalar sum = 0;
        
        for (int i = 0; i < height * width; ++i)
        {{
            int idx = i * channels + c;
            sum += inputs[idx];
        }}
        Scalar mean = sum / (height * width);
        Scalar var = 0;
        
        for (int i = 0; i < height * width; ++i)
        {{
            int idx = i * channels + c;
            var += (inputs[idx] - mean) * (inputs[idx] - mean);
        }}
        var /= (height * width);
        
        for (int i = 0; i < height * width; ++i)
        {{
            int idx = i * channels + c;
            outputs[idx] = gamma[c] * ((inputs[idx] - mean) / std::sqrt(var + epsilon)) + beta[c];
        }}
    }}
}}
""",
        "UnitNormalization": f"""
template <typename Scalar, int size>
inline void UnitNormalization_{base_file_name}(Scalar * __restrict outputs,
                              const Scalar * __restrict inputs,
                              Scalar epsilon) noexcept
{{
    Scalar sum_sq = 0;
    for (int i = 0; i < size; ++i) 
    {{
        sum_sq += inputs[i] * inputs[i];
    }}
    Scalar inv_norm = Scalar(1) / std::sqrt(sum_sq + epsilon);
    for (int i = 0; i < size; ++i) 
    {{
        outputs[i] = inputs[i] * inv_norm;
    }}
}}
""",
    }

    # convolution functions
    convolution_functions = {
        "Conv1D": f"""
template <typename Scalar, int out_size, typename ActFun>
inline void Conv1D_{base_file_name}(Scalar * __restrict outputs, const Scalar * __restrict inputs, const Scalar * __restrict weights, const Scalar * __restrict biases,
                   int in_size, int kernel_size, int stride, int padding,
                   ActFun activation_function, Scalar alpha) noexcept
{{
    for (int o = 0; o < out_size; ++o)
    {{
        Scalar sum = 0;
        
        for (int k = 0; k < kernel_size; ++k)
        {{
            int in_index = o * stride - padding + k;
            if (in_index >= 0 && in_index < in_size)
            {{
                int weight_index = k * out_size + o;
                sum += inputs[in_index] * weights[weight_index];
            }}
        }}
        sum += biases[o];
        activation_function(outputs[o], sum, alpha);
    }}
}}
""",
        "Conv1DTranspose": f"""
template <typename Scalar, int out_channels, int out_length, typename ActFun>
inline void Conv1DTranspose_{base_file_name}(Scalar * __restrict outputs, const Scalar * __restrict inputs, const Scalar * __restrict kernels, const Scalar * __restrict biases, int in_channels,int in_length, int kernel_width, int stride, int padding, ActFun activation_function, Scalar alpha)
{{
    for (int ow = 0; ow < out_length; ++ow)
    {{
        
        for (int oc = 0; oc < out_channels; ++oc)
        {{
            int idx = ow * out_channels + oc;
            outputs[idx] = biases ? biases[oc] : Scalar(0);
        }}
    }}

    for (int ic = 0; ic < in_channels; ++ic)
    {{
        for (int iw = 0; iw < in_length; ++iw)
        {{
            Scalar val = inputs[iw * in_channels + ic];
            int base = iw * stride - padding;

            for (int k = 0; k < kernel_width; ++k)
            {{
                int ow = base + k;
                if (ow < 0 || ow >= out_length)
                    continue;

                int fk = kernel_width - 1 - k;
                int ker_base = fk * (out_channels * in_channels) + ic;
                int out_base = ow * out_channels;
                
                for (int oc = 0; oc < out_channels; ++oc)
                {{
                    int k_idx = ker_base + oc * in_channels;
                    outputs[out_base + oc] += val * kernels[k_idx];
                }}
            }}
        }}
    }}

    int total = out_length * out_channels;
    for (int i = 0; i < total; ++i)
    {{
        activation_function(outputs[i], outputs[i], alpha);
    }}

    /*
    NEEDED TO REORDER OUTPUTS (WILL UPDATE)
    */
    std::array<Scalar, out_length * out_channels> tmp;
    std::copy(outputs, outputs + total, tmp.begin());
    for (int ow = 0; ow < out_length; ++ow)
    {{
        int dst = ow * out_channels;
        int src = (out_length - 1 - ow) * out_channels;
        
        for (int oc = 0; oc < out_channels; ++oc)
        {{
            outputs[dst + oc] = tmp[src + oc];
        }}
    }}
    /*
    NEEDED TO REORDER OUTPUTS (WILL UPDATE)
    */
   
}}
""",
        "Conv2D": f"""
template <typename Scalar, int out_channels, int out_height, int out_width, typename ActivationFunc>
inline void Conv2D_{base_file_name}(Scalar * __restrict outputs, const Scalar * __restrict inputs, const Scalar *__restrict weights, const Scalar *__restrict biases, int in_channels, int in_height, int in_width, int kernel_height, int kernel_width, int stride_height, int stride_width, int padding_height, int padding_width, ActivationFunc activation_function, Scalar alpha) noexcept
{{
 
    Scalar sum_buf[out_channels];
    const int input_row_stride = in_width * in_channels;
    const int weights_per_khkw = in_channels * out_channels;  
    const int weights_per_kh = kernel_width * weights_per_khkw;

    for (int oh = 0; oh < out_height; ++oh)
    {{
        const int h_origin = oh * stride_height - padding_height;

        for (int ow = 0; ow < out_width; ++ow)
        {{
            for (int oc = 0; oc < out_channels; ++oc) {{
                sum_buf[oc] = biases[oc];
            }}
            const int w_origin = ow * stride_width - padding_width;

            const int kh_min = std::max(0,       -h_origin);
            const int kh_max = std::min(kernel_height, in_height - h_origin);
            const int kw_min = std::max(0,       -w_origin);
            const int kw_max = std::min(kernel_width, in_width - w_origin);

            for (int kh = kh_min; kh < kh_max; ++kh)
            {{
                const int in_h = h_origin + kh;
                const int input_row_offset = in_h * input_row_stride;
                const int weight_kh_offset = kh * weights_per_kh;

                for (int kw = kw_min; kw < kw_max; ++kw)
                {{
                    const int in_w = w_origin + kw;
                    const int input_base = input_row_offset + in_w * in_channels;

                    const int weight_base = weight_kh_offset + (kw * weights_per_khkw);

                    for (int ic = 0; ic < in_channels; ++ic)
                    {{
                        const Scalar input_val = inputs[input_base + ic];
                        const Scalar *w_ptr = weights + weight_base + ic * out_channels;

                        
                        for (int oc = 0; oc < out_channels; ++oc) {{
                            sum_buf[oc] += input_val * w_ptr[oc];
                        }}
                    }}
                }}
            }}

            Scalar *out_pixel_ptr = outputs + ((oh * out_width + ow) * out_channels);
            for (int oc = 0; oc < out_channels; ++oc) {{
                activation_function(out_pixel_ptr[oc], sum_buf[oc], alpha);
            }}
        }}
    }}
}}
""",
        "Conv2DTranspose": f"""
template <typename Scalar, int out_channels, int out_height, int out_width, typename ActFun>
inline void Conv2DTranspose_{base_file_name}(Scalar * __restrict outputs, const Scalar * __restrict inputs,
                     const Scalar * __restrict kernels, const Scalar * __restrict biases,
                     int in_channels, int in_height, int in_width,
                     int kernel_height, int kernel_width, int stride_height,
                     int stride_width, int padding_height, int padding_width,
                     ActFun activation_function,
                     Scalar alpha)
{{
    for (int h = 0; h < out_height; ++h)
    {{
        for (int w = 0; w < out_width; ++w)
        {{
            
            for (int oc = 0; oc < out_channels; ++oc)
            {{
                int idx = (h * out_width + w) * out_channels + oc;
                outputs[idx] = biases ? biases[oc] : Scalar(0);
            }}
        }}
    }}

    for (int ic = 0; ic < in_channels; ++ic)
    {{
        for (int ih = 0; ih < in_height; ++ih)
        {{
            for (int iw = 0; iw < in_width; ++iw)
            {{
                Scalar current_value = inputs[(ih * in_width + iw) * in_channels + ic];
                int base_height = ih * stride_height - padding_height;
                int base_width = iw * stride_width - padding_width;

                for (int kh = 0; kh < kernel_height; ++kh)
                {{
                    int oh = base_height + kh;
                    if (oh < 0 || oh >= out_height)
                        continue;
                    for (int kw = 0; kw < kernel_width; ++kw)
                    {{
                        int ow = base_width + kw;
                        if (ow < 0 || ow >= out_width)
                            continue;
                        int out_base = (oh * out_width + ow) * out_channels;
                        int fh = kernel_height - 1 - kh;
                        int fw = kernel_width - 1 - kw;
                        int ker_base =
                            (fh * kernel_width + fw) * (out_channels * in_channels) + ic;
                        
                        for (int oc = 0; oc < out_channels; ++oc)
                        {{
                            int k_idx = ker_base + oc * in_channels;
                            outputs[out_base + oc] += current_value * kernels[k_idx];
                        }}
                    }}
                }}
            }}
        }}
    }}

    int total = out_height * out_width * out_channels;
    for (int i = 0; i < total; ++i)
    {{
        activation_function(outputs[i], outputs[i], alpha);
    }}

    std::array<Scalar, out_height * out_width * out_channels> tmp;
    std::copy(outputs, outputs + total, tmp.begin());

    /*
    NEEDED TO REORDER OUTPUTS (WILL UPDATE)
    */
    for (int h = 0; h < out_height; ++h)
        for (int w = 0; w < out_width; ++w)
        {{
            const int dst_base = (h * out_width + w) * out_channels;
            const int src_base =
                ((out_height - 1 - h) * out_width + (out_width - 1 - w)) *
                out_channels;

            
            for (int c = 0; c < out_channels; ++c)
                outputs[dst_base + c] = tmp[src_base + c];
        }}
    /*
    NEEDED TO REORDER OUTPUTS (WILL UPDATE)
    */
}}
""",
        "Conv3D": f"""
template <typename Scalar, int out_channels, int out_depth, int out_height, int out_width, typename ActFun>
inline void Conv3D_{base_file_name}(Scalar * __restrict outputs, const Scalar * __restrict inputs, const Scalar * __restrict weights, const Scalar * __restrict biases,
                   int in_channels, int in_depth, int in_height, int in_width,
                   int kernel_depth, int kernel_height, int kernel_width, int stride_depth, int stride_height, int stride_width,
                   int padding_depth, int padding_height, int padding_width,
                   ActFun activation_function, Scalar alpha) noexcept
{{
    for (int oc = 0; oc < out_channels; ++oc)
    {{
        for (int od = 0; od < out_depth; ++od)
        {{
            for (int oh = 0; oh < out_height; ++oh)
            {{
                for (int ow = 0; ow < out_width; ++ow)
                {{
                    Scalar sum = 0;
                    for (int ic = 0; ic < in_channels; ++ic)
                    {{
                        for (int kd = 0; kd < kernel_depth; ++kd)
                        {{
                            for (int kh = 0; kh < kernel_height; ++kh)
                            {{
                                
                                for (int kw = 0; kw < kernel_width; ++kw)
                                {{
                                    int in_d = od * stride_depth - padding_depth + kd;
                                    int in_h = oh * stride_height - padding_height + kh;
                                    int in_w = ow * stride_width - padding_width + kw;
                                    if (in_d >= 0 && in_d < in_depth &&
                                        in_h >= 0 && in_h < in_height &&
                                        in_w >= 0 && in_w < in_width)
                                    {{
                                        int input_index = ((in_d * in_height * in_width * in_channels) +
                                                           (in_h * in_width * in_channels) +
                                                           (in_w * in_channels) + ic);
                                        int weight_index = (((((kd * kernel_height + kh) * kernel_width + kw) * in_channels + ic) * out_channels) + oc);
                                        sum += inputs[input_index] * weights[weight_index];
                                    }}
                                }}
                            }}
                        }}
                    }}
                    sum += biases[oc];
                    int output_index = ((od * out_height * out_width * out_channels) +
                                        (oh * out_width * out_channels) +
                                        (ow * out_channels) + oc);
                    activation_function(outputs[output_index], sum, alpha);
                }}
            }}
        }}
    }}
}}
""",
        "Conv3DTranspose": f"""
template <typename Scalar, int out_channels, int out_depth, int out_height,
          int out_width, typename ActFun>
inline void Conv3DTranspose_{base_file_name}(Scalar * __restrict outputs, const Scalar * __restrict inputs,
                     const Scalar * __restrict kernels, const Scalar * __restrict biases,
                     int in_channels, int in_depth, int in_height, int in_width,
                     int kernel_depth, int kernel_height, int kernel_width,
                     int stride_depth, int stride_height, int stride_width,
                     int padding_depth, int padding_height, int padding_width,
                     ActFun activation_function,
                     Scalar alpha)
{{
    for (int d = 0; d < out_depth; ++d)
    {{
        for (int h = 0; h < out_height; ++h)
        {{
            for (int w = 0; w < out_width; ++w)
            {{
                
                for (int oc = 0; oc < out_channels; ++oc)
                {{
                    int idx = ((d * out_height + h) * out_width + w) * out_channels + oc;
                    outputs[idx] = biases ? biases[oc] : Scalar(0);
                }}
            }}
        }}
    }}

    for (int ic = 0; ic < in_channels; ++ic)
    {{
        for (int id = 0; id < in_depth; ++id)
        {{
            for (int ih = 0; ih < in_height; ++ih)
            {{
                for (int iw = 0; iw < in_width; ++iw)
                {{
                    Scalar in_val =
                        inputs[((id * in_height + ih) * in_width + iw) * in_channels +
                               ic];
                    int base_d = id * stride_depth - padding_depth;
                    int base_h = ih * stride_height - padding_height;
                    int base_w = iw * stride_width - padding_width;

                    for (int kd = 0; kd < kernel_depth; ++kd)
                    {{
                        int od = base_d + kd;
                        if (od < 0 || od >= out_depth)
                            continue;
                        for (int kh = 0; kh < kernel_height; ++kh)
                        {{
                            int oh = base_h + kh;
                            if (oh < 0 || oh >= out_height)
                                continue;
                            for (int kw = 0; kw < kernel_width; ++kw)
                            {{
                                int ow = base_w + kw;
                                if (ow < 0 || ow >= out_width)
                                    continue;

                                int out_base =
                                    ((od * out_height + oh) * out_width + ow) * out_channels;
                                int fd = kernel_depth - 1 - kd;
                                int fh = kernel_height - 1 - kh;
                                int fw = kernel_width - 1 - kw;
                                int ker_base = ((fd * kernel_height + fh) * kernel_width + fw) *
                                                   (out_channels * in_channels) +
                                               ic;

                                
                                for (int oc = 0; oc < out_channels; ++oc)
                                {{
                                    int k_idx = ker_base + oc * in_channels;
                                    outputs[out_base + oc] += in_val * kernels[k_idx];
                                }}
                            }}
                        }}
                    }}
                }}
            }}
        }}
    }}

    int total = out_depth * out_height * out_width * out_channels;
    for (int i = 0; i < total; ++i)
    {{
        activation_function(outputs[i], outputs[i], alpha);
    }}

    /*
    NEEDED TO REORDER OUTPUTS (WILL UPDATE)
    */
    std::array<Scalar, out_depth * out_height * out_width * out_channels> tmp;
    std::copy(outputs, outputs + total, tmp.begin());

    for (int d = 0; d < out_depth; ++d)
    {{
        for (int h = 0; h < out_height; ++h)
        {{
            for (int w = 0; w < out_width; ++w)
            {{
                int dst_base = ((d * out_height + h) * out_width + w) * out_channels;
                int src_base =
                    (((out_depth - 1 - d) * out_height + (out_height - 1 - h)) *
                         out_width +
                     (out_width - 1 - w)) *
                    out_channels;
                
                for (int c = 0; c < out_channels; ++c)
                {{
                    outputs[dst_base + c] = tmp[src_base + c];
                }}
            }}
        }}
    }}
    /*
    NEEDED TO REORDER OUTPUTS (WILL UPDATE)
    */
   
}}
""",
        "DepthwiseConv2D": f"""
template <typename Scalar, typename ActFun>
inline void DepthwiseConv2D_{base_file_name}(Scalar * __restrict outputs, const Scalar * __restrict inputs, const Scalar * __restrict weights, const Scalar * __restrict biases,
                            int out_channels, int out_height, int out_width,
                            int in_channels, int in_height, int in_width,
                            int kernel_height, int kernel_width, int stride_height, int stride_width,
                            int padding_height, int padding_width,
                            ActFun activation_function, Scalar alpha) noexcept
{{
    for (int c = 0; c < in_channels; ++c)
    {{
        for (int oh = 0; oh < out_height; ++oh)
        {{
            for (int ow = 0; ow < out_width; ++ow)
            {{
                Scalar sum = 0;
                for (int kh = 0; kh < kernel_height; ++kh)
                {{
                    
                    for (int kw = 0; kw < kernel_width; ++kw)
                    {{
                        int in_h = oh * stride_height - padding_height + kh;
                        int in_w = ow * stride_width - padding_width + kw;
                        if (in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width)
                        {{
                            int input_index = (in_h * in_width * in_channels) + (in_w * in_channels) + c;
                            int weight_index = (kh * kernel_width + kw) * in_channels + c;
                            sum += inputs[input_index] * weights[weight_index];
                        }}
                    }}
                }}
                sum += biases[c];
                int output_index = (oh * out_width * in_channels) + (ow * in_channels) + c;
                activation_function(outputs[output_index], sum, alpha);
            }}
        }}
    }}
}}
""",
        "SeparableConv2D": f"""
template <typename Scalar>
inline void DepthwiseForsSeparableConv2D(Scalar *__restrict outputs, const Scalar *__restrict inputs, const Scalar *__restrict weights, const Scalar *__restrict biases,
                                  int out_height, int out_width,
                                  int in_channels, int in_height, int in_width,
                                  int kernel_height, int kernel_width, int stride_height, int stride_width,
                                  int padding_height, int padding_width) noexcept
{{
    for (int c = 0; c < in_channels; ++c)
    {{
        for (int oh = 0; oh < out_height; ++oh)
        {{
            for (int ow = 0; ow < out_width; ++ow)
            {{
                Scalar sum = 0;
                for (int kh = 0; kh < kernel_height; ++kh)
                {{
                    
                    for (int kw = 0; kw < kernel_width; ++kw)
                    {{
                        int in_h = oh * stride_height - padding_height + kh;
                        int in_w = ow * stride_width - padding_width + kw;
                        if (in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width)
                        {{
                            int input_index = (in_h * in_width * in_channels) + (in_w * in_channels) + c;
                            int weight_index = (kh * kernel_width + kw) * in_channels + c;
                            sum += inputs[input_index] * weights[weight_index];
                        }}
                    }}
                }}
                sum += biases[c];
                int output_index = ((oh * out_width + ow) * in_channels) + c;
                outputs[output_index] = sum;
            }}
        }}
    }}
}}

template <typename Scalar, int out_channels, int out_height, int out_width, typename ActFun>
inline void SeparableConv2D_{base_file_name}(
    Scalar *__restrict outputs,
    const Scalar *__restrict inputs,
    const Scalar *__restrict depthwise_weights,
    const Scalar *__restrict pointwise_weights,
    const Scalar *__restrict biases,
    int in_channels, int in_height, int in_width,
    int kernel_height, int kernel_width,
    int stride_height, int stride_width,
    int padding_height, int padding_width,
    ActFun activation_function, Scalar alpha) noexcept
{{
    std::vector<Scalar> depthwise_output(out_height * out_width * in_channels, 0);
    std::vector<Scalar> zero_bias(in_channels, 0);
    DepthwiseForsSeparableConv2D(
        depthwise_output.data(), inputs, depthwise_weights, zero_bias.data(), out_height, out_width,
        in_channels, in_height, in_width,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_height, padding_width);
    for (int oc = 0; oc < out_channels; ++oc)
    {{
        for (int i = 0; i < out_height * out_width; ++i)
        {{
            Scalar sum = 0;
            
            for (int ic = 0; ic < in_channels; ++ic)
            {{
                int index = i * in_channels + ic;
                int weight_index = ic * out_channels + oc;
                sum += depthwise_output[index] * pointwise_weights[weight_index];
            }}
            sum += biases[oc];
            int output_index = i * out_channels + oc;
            activation_function(outputs[output_index], sum, alpha);
        }}
    }}
}}
""",
        "ConvLSTM2D": f"""
template<typename Scalar, typename ActFun>
inline void ConvLSTM2D_{base_file_name}(Scalar* __restrict outputs,
                       const Scalar* __restrict inputs,
                       const Scalar* __restrict kernel,
                       const Scalar* __restrict recurrent_kernel,
                       const Scalar* __restrict bias,
                       int time_steps,
                       int in_channels,
                       int in_height,
                       int in_width,
                       int filters,
                       int kernel_height,
                       int kernel_width,
                       int stride_height,
                       int stride_width,
                       int padding_height,
                       int padding_width,
                       ActFun activation_function,
                       activationFunction<Scalar> recurrent_activation_function,
                       Scalar alpha) noexcept
{{
    // hidden + cell state buffers
    std::vector<Scalar> h_state(filters * in_height * in_width, Scalar(0));
    std::vector<Scalar> c_state(filters * in_height * in_width, Scalar(0));
    int spatial_size = in_height * in_width;

    for (int t = 0; t < time_steps; ++t) {{
        // slice for time step t
        const Scalar* x_t = inputs + t * in_channels * spatial_size;

        // gate buffers
        std::vector<Scalar> i_gate(filters * spatial_size, 0);
        std::vector<Scalar> f_gate(filters * spatial_size, 0);
        std::vector<Scalar> g_gate(filters * spatial_size, 0);
        std::vector<Scalar> o_gate(filters * spatial_size, 0);

        // --- Input convolution for all 4 gates ---
        for (int oc = 0; oc < filters; ++oc) {{
            for (int oh = 0; oh < in_height; ++oh) {{
                for (int ow = 0; ow < in_width; ++ow) {{
                    int out_idx = oc + filters * (ow + in_width * oh);
                    Scalar sum_i = 0, sum_f = 0, sum_g = 0, sum_o = 0;
                    for (int ic = 0; ic < in_channels; ++ic) {{
                        for (int kh = 0; kh < kernel_height; ++kh) {{
                            
                            for (int kw = 0; kw < kernel_width; ++kw) {{
                                int ih = oh * stride_height - padding_height + kh;
                                int iw = ow * stride_width  - padding_width  + kw;
                                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {{
                                    int in_idx = ic + in_channels * (iw + in_width * ih);
                                    int k_base = ((kh * kernel_width + kw) * in_channels + ic);
                                    // kernel layout: [kh,kw,in_channels,4*filters]
                                    sum_i += x_t[in_idx] * kernel[(k_base * 4 + 0) * filters + oc];
                                    sum_f += x_t[in_idx] * kernel[(k_base * 4 + 1) * filters + oc];
                                    sum_g += x_t[in_idx] * kernel[(k_base * 4 + 2) * filters + oc];
                                    sum_o += x_t[in_idx] * kernel[(k_base * 4 + 3) * filters + oc];
                                }}
                            }}
                        }}
                    }}
                    i_gate[out_idx] = sum_i;
                    f_gate[out_idx] = sum_f;
                    g_gate[out_idx] = sum_g;
                    o_gate[out_idx] = sum_o;
                }}
            }}
        }}

        // --- Recurrent convolution and bias add ---
        for (int oc = 0; oc < filters; ++oc) {{
            for (int oh = 0; oh < in_height; ++oh) {{
                for (int ow = 0; ow < in_width; ++ow) {{
                    int idx = oc + filters * (ow + in_width * oh);
                    Scalar rec_i = 0, rec_f = 0, rec_g = 0, rec_o = 0;
                    for (int kc = 0; kc < filters; ++kc) {{
                        for (int kh = 0; kh < kernel_height; ++kh) {{
                            
                            for (int kw = 0; kw < kernel_width; ++kw) {{
                                int ih = oh * stride_height - padding_height + kh;
                                int iw = ow * stride_width  - padding_width  + kw;
                                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {{
                                    int h_idx = kc + filters * (iw + in_width * ih);
                                    int rk_base = ((kh * kernel_width + kw) * filters + kc);
                                    rec_i += h_state[h_idx] * recurrent_kernel[(rk_base * 4 + 0) * filters + oc];
                                    rec_f += h_state[h_idx] * recurrent_kernel[(rk_base * 4 + 1) * filters + oc];
                                    rec_g += h_state[h_idx] * recurrent_kernel[(rk_base * 4 + 2) * filters + oc];
                                    rec_o += h_state[h_idx] * recurrent_kernel[(rk_base * 4 + 3) * filters + oc];
                                }}
                            }}
                        }}
                    }}
                    i_gate[idx] += rec_i + bias[oc];
                    f_gate[idx] += rec_f + bias[filters + oc];
                    g_gate[idx] += rec_g + bias[2 * filters + oc];
                    o_gate[idx] += rec_o + bias[3 * filters + oc];
                }}
            }}
        }}

        // --- Activation + state update + write output ---
        for (int idx = 0; idx < filters * spatial_size; ++idx) {{
            recurrent_activation_function(i_gate[idx], i_gate[idx], alpha);
            recurrent_activation_function(f_gate[idx], f_gate[idx], alpha);
            activation_function(g_gate[idx], g_gate[idx], alpha);
            recurrent_activation_function(o_gate[idx], o_gate[idx], alpha);

            // cell state
            c_state[idx] = f_gate[idx] * c_state[idx] + i_gate[idx] * g_gate[idx];
            // hidden state
            Scalar c_act;
            activation_function(c_act, c_state[idx], alpha);
            h_state[idx] = o_gate[idx] * c_act;

            // write out
            outputs[t * filters * spatial_size + idx] = h_state[idx];
        }}
    }}
}}
""",
    }

    pooling_functions = {
        "MaxPooling1D": f"""
template <typename Scalar, int pool_size, int stride>
inline void MaxPooling1D_{base_file_name}(Scalar * __restrict outputs, const Scalar * __restrict inputs, int in_length, int channels) noexcept
{{
    int out_length = (in_length - pool_size) / stride + 1;
    for (int c = 0; c < channels; ++c)
    {{
        for (int o = 0; o < out_length; ++o)
        {{
            Scalar max_val = inputs[(o * stride * channels) + c];
            
            for (int p = 0; p < pool_size; ++p)
            {{
                int idx = ((o * stride + p) * channels) + c;
                if (inputs[idx] > max_val)
                {{
                    max_val = inputs[idx];
                }}
            }}
            outputs[o * channels + c] = max_val;
        }}
    }}
}}
""",
        "MaxPooling2D": f"""
template <typename Scalar, int pool_height, int pool_width, int stride_h, int stride_w>
inline void MaxPooling2D_{base_file_name}(Scalar * __restrict outputs, const Scalar * __restrict inputs, int in_height, int in_width, int channels) noexcept
{{
    int out_height = (in_height - pool_height) / stride_h + 1;
    int out_width = (in_width - pool_width) / stride_w + 1;
    for (int c = 0; c < channels; ++c)
    {{
        for (int oh = 0; oh < out_height; ++oh)
        {{
            for (int ow = 0; ow < out_width; ++ow)
            {{
                Scalar max_val = -std::numeric_limits<Scalar>::infinity();
                for (int ph = 0; ph < pool_height; ++ph)
                {{
                    
                    for (int pw = 0; pw < pool_width; ++pw)
                    {{
                        int in_h = oh * stride_h + ph;
                        int in_w = ow * stride_w + pw;
                        int idx = (in_h * in_width * channels) + (in_w * channels) + c;
                        if (inputs[idx] > max_val)
                        {{
                            max_val = inputs[idx];
                        }}
                    }}
                }}
                int out_idx = (oh * out_width * channels) + (ow * channels) + c;
                outputs[out_idx] = max_val;
            }}
        }}
    }}
}}
""",
        "MaxPooling3D": f"""
template <typename Scalar, int pool_depth, int pool_height, int pool_width, int stride_depth, int stride_height, int stride_width>
inline void MaxPooling3D_{base_file_name}(Scalar * __restrict outputs, const Scalar * __restrict inputs, int in_depth, int in_height, int in_width, int channels) noexcept
{{
    int out_depth = (in_depth - pool_depth) / stride_depth + 1;
    int out_height = (in_height - pool_height) / stride_height + 1;
    int out_width = (in_width - pool_width) / stride_width + 1;
    for (int c = 0; c < channels; ++c) {{
        for (int d = 0; d < out_depth; ++d) {{
            for (int h = 0; h < out_height; ++h) {{
                for (int w = 0; w < out_width; ++w) {{
                    Scalar max_val = inputs[(((d * stride_depth * in_height + h * stride_height) * in_width + w * stride_width) * channels) + c];
                    for (int pd = 0; pd < pool_depth; ++pd) {{
                        for (int ph = 0; ph < pool_height; ++ph) {{
                            
                            for (int pw = 0; pw < pool_width; ++pw) {{
                                int idx = ((((d * stride_depth + pd) * in_height + (h * stride_height + ph)) * in_width + (w * stride_width + pw)) * channels) + c;
                                if (inputs[idx] > max_val) {{
                                    max_val = inputs[idx];
                                }}
                            }}
                        }}
                    }}
                    outputs[(((d * out_height + h) * out_width + w) * channels) + c] = max_val;
                }}
            }}
        }}
    }}
}}
""",
        "AvgPooling1D": f"""
template <typename Scalar, int pool_size, int stride>
inline void AvgPooling1D_{base_file_name}(Scalar * __restrict outputs, const Scalar * __restrict inputs, int in_length, int channels) noexcept
{{
    int out_length = (in_length - pool_size) / stride + 1;
    for (int c = 0; c < channels; ++c)
    {{
        for (int o = 0; o < out_length; ++o)
        {{
            Scalar sum = 0;
            
            for (int p = 0; p < pool_size; ++p)
            {{
                int idx = ((o * stride + p) * channels) + c;
                sum += inputs[idx];
            }}
            outputs[o * channels + c] = sum / pool_size;
        }}
    }}
}}
""",
        "AvgPooling2D": f"""
template <typename Scalar, int pool_height, int pool_width, int stride_h, int stride_w>
inline void AvgPooling2D_{base_file_name}(Scalar * __restrict outputs, const Scalar * __restrict inputs, int in_height, int in_width, int channels) noexcept
{{
    int out_height = (in_height - pool_height) / stride_h + 1;
    int out_width = (in_width - pool_width) / stride_w + 1;
    for (int c = 0; c < channels; ++c)
    {{
        for (int oh = 0; oh < out_height; ++oh)
        {{
            for (int ow = 0; ow < out_width; ++ow)
            {{
                Scalar sum = 0;
                for (int ph = 0; ph < pool_height; ++ph)
                {{
                    
                    for (int pw = 0; pw < pool_width; ++pw)
                    {{
                        int in_h = oh * stride_h + ph;
                        int in_w = ow * stride_w + pw;
                        int idx = (in_h * in_width * channels) + (in_w * channels) + c;
                        sum += inputs[idx];
                    }}
                }}
                int out_idx = (oh * out_width * channels) + (ow * channels) + c;
                outputs[out_idx] = sum / (pool_height * pool_width);
            }}
        }}
    }}
}}
""",
        "AvgPooling3D": f"""
template <typename Scalar, int pool_depth, int pool_height, int pool_width, int stride_depth, int stride_height, int stride_width>
inline void AvgPooling3D_{base_file_name}(Scalar * __restrict outputs, const Scalar * __restrict inputs, int in_depth, int in_height, int in_width, int channels) noexcept
{{
    int out_depth = (in_depth - pool_depth) / stride_depth + 1;
    int out_height = (in_height - pool_height) / stride_height + 1;
    int out_width = (in_width - pool_width) / stride_width + 1;
    for (int c = 0; c < channels; ++c) {{
        for (int d = 0; d < out_depth; ++d) {{
            for (int h = 0; h < out_height; ++h) {{
                for (int w = 0; w < out_width; ++w) {{
                    Scalar sum = 0;
                    for (int pd = 0; pd < pool_depth; ++pd) {{
                        for (int ph = 0; ph < pool_height; ++ph) {{
                            
                            for (int pw = 0; pw < pool_width; ++pw) {{
                                int idx = ((((d * stride_depth + pd) * in_height + (h * stride_height + ph)) * in_width + (w * stride_width + pw)) * channels) + c;
                                sum += inputs[idx];
                            }}
                        }}
                    }}
                    outputs[(((d * out_height + h) * out_width + w) * channels) + c] = sum / (pool_depth * pool_height * pool_width);
                }}
            }}
        }}
    }}
}}   
""",
        "GlobalMaxPooling1D": f"""
template <typename Scalar>
inline void GlobalMaxPooling1D_{base_file_name}(Scalar * __restrict outputs, const Scalar * __restrict inputs, int in_length, int channels) noexcept
{{
    for (int c = 0; c < channels; ++c)
    {{
        Scalar max_val = -std::numeric_limits<Scalar>::infinity();
        
        for (int i = 0; i < in_length; ++i)
        {{
            int idx = (i * channels) + c;
            if (inputs[idx] > max_val)
            {{
                max_val = inputs[idx];
            }}
        }}
        output[c] = max_val;
    }}
}}
""",
        "GlobalMaxPooling2D": f"""
template <typename Scalar>
inline void GlobalMaxPooling2D_{base_file_name}(Scalar * __restrict outputs, const Scalar * __restrict inputs, int in_height, int in_width, int channels) noexcept
{{
    for (int c = 0; c < channels; ++c)
    {{
        Scalar max_val = -std::numeric_limits<Scalar>::infinity();
        for (int h = 0; h < in_height; ++h)
        {{
            
            for (int w = 0; w < in_width; ++w)
            {{
                int idx = (h * in_width * channels) + (w * channels) + c;
                if (inputs[idx] > max_val)
                {{
                    max_val = inputs[idx];
                }}
            }}
        }}
        output[c] = max_val;
    }}
}}
""",
        "GlobalMaxPooling3D": f"""
template <typename Scalar>
inline void GlobalMaxPooling3D_{base_file_name}(Scalar * __restrict outputs, const Scalar * __restrict inputs, int in_depth, int in_height, int in_width, int channels) noexcept
{{
    for (int c = 0; c < channels; ++c)
    {{
        Scalar max_val = -std::numeric_limits<Scalar>::infinity();
        for (int d = 0; d < in_depth; ++d)
        {{
            for (int h = 0; h < in_height; ++h)
            {{
                
                for (int w = 0; w < in_width; ++w)
                {{
                    int idx = ((d * in_height * in_width * channels) + (h * in_width * channels) + (w * channels) + c);
                    if (inputs[idx] > max_val)
                    {{
                        max_val = inputs[idx];
                    }}
                }}
            }}
        }}
        output[c] = max_val;
    }}
}}
""",
        "GlobalAvgPooling1D": f"""
template <typename Scalar>
inline void GlobalAvgPooling1D_{base_file_name}(Scalar * __restrict outputs, const Scalar * __restrict inputs, int in_length, int channels)
{{
    for (int ic = 0, ic < channels, ic++)
    {{
        
        for (int i = 0, i < in_length, i++)
        {{
            int idx = (i * channels) + (channels) + c;
            sum += inputs[idx];
        }}
        output[c] = sum / (length);
    }}
}}
""",
        "GlobalAvgPooling2D": f"""
template <typename Scalar>
inline void GlobalAvgPooling2D_{base_file_name}(Scalar * __restrict outputs, const Scalar * __restrict inputs, int in_height, int in_width, int channels) noexcept
{{
    // Compute global average per channel.
    for (int c = 0; c < channels; ++c)
    {{
        Scalar sum = 0;
        for (int h = 0; h < in_height; ++h)
        {{
            
            for (int w = 0; w < in_width; ++w)
            {{
                int idx = (h * in_width * channels) + (w * channels) + c;
                sum += inputs[idx];
            }}
        }}
        output[c] = sum / (in_height * in_width);
    }}
}}
""",
        "GlobalAvgPooling3D": f"""
template <typename Scalar>
inline void GlobalAvgPooling3D_{base_file_name}(Scalar * __restrict outputs, const Scalar * __restrict inputs, int in_depth, int in_height, int in_width, int channels) noexcept
{{
    for (int c = 0; c < channels; ++c)
    {{
        Scalar sum = 0;
        for (int d = 0; d < in_depth; ++d)
        {{
            for (int h = 0; h < in_height; ++h)
            {{
                
                for (int w = 0; w < in_width; ++w)
                {{
                    int idx = ((d * in_height * in_width * channels) + (h * in_width * channels) + (w * channels) + c);
                    sum += inputs[idx];
                }}
            }}
        }}
        output[c] = sum / (in_depth * in_height * in_width);
    }}
}}
""",
    }

    try:
        # set every function and append it to cpp_code
        current_activations = set(activation_functions)
        current_activations = {
            ("tanhCustom" if act == "tanh" else act)
            for act in current_activations
            if act is not None and act != "Activation"
        }

        cpp_lambda = """"""

        # set activation functions
        for act in current_activations:
            if act in lambda_functions:
                cpp_lambda += lambda_functions[act]

        # deduplicate layer_type list
        unique_layer_types = {lt for lt in layer_type if lt is not None}

        # set layer propagation layers
        for type in unique_layer_types:
            if type in preprocessing_functions:
                cpp_code += preprocessing_functions[type]
            if type in dense_function:
                cpp_code += dense_function[type]
            if type in reshape_functions:
                cpp_code += reshape_functions[type]
            if type in normalization_functions:
                cpp_code += normalization_functions[type]
            if type in convolution_functions:
                cpp_code += convolution_functions[type]
            if type in pooling_functions:
                cpp_code += pooling_functions[type]
    except ValueError as e:
        print(
            f"\nError in setting layer propagation functions --> ",
            e,
        )

    return cpp_code, cpp_lambda
