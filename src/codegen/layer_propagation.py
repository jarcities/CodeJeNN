import os
import absl.logging
import warnings

absl.logging.set_verbosity("error")
warnings.filterwarnings("ignore", category=UserWarning, module="keras")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def layer_propagation(cpp_code, activation_functions, layer_type):
    """
    Generate C++ lambda-based activation functions (with no indentation for the lambdas)
    and normalization functions. ForwardPass also remains as Code 2 style.
    """

    # regular forward pass
    dense_function = {
        "Dense": """
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
"""
    }
    
    # lambda activation functions
    lambda_functions = {
        "relu": """
    auto relu = +[](Scalar& output, Scalar input, Scalar alpha) noexcept {
        output = input > 0 ? input : 0;
    };
""",
        "sigmoid": """
    auto sigmoid = +[](Scalar& output, Scalar input, Scalar alpha) noexcept {
        output = 1 / (1 + std::exp(-input));
    };
""",
        "tanhCustom": """
    auto tanhCustom = +[](Scalar& output, Scalar input, Scalar alpha) noexcept {
        output = std::tanh(input);
    };
""",
        "leakyrelu": """
    auto leakyrelu = +[](Scalar& output, Scalar input, Scalar alpha) noexcept {
        output = input > 0 ? input : alpha * input;
    };
""",
        "linear": """
    auto linear = +[](Scalar& output, Scalar input, Scalar alpha) noexcept {
        output = input;
    };
""",
        "elu": """
    auto elu = +[](Scalar& output, Scalar input, Scalar alpha) noexcept {
        output = input > 0 ? input : alpha * (std::exp(input) - 1);
    };
""",
        "selu": """
    auto selu = +[](Scalar& output, Scalar input, Scalar alpha) noexcept {
        output = 1.0507009873554804934193349852946 * (input > 0 ? input : 1.6732632423543772848170429916717 * (std::exp(input) - 1));
    };
""",
        "swish": """
    auto swish = +[](Scalar& output, Scalar input, Scalar alpha) noexcept {
        output = input / (1 + std::exp(-alpha * input));
    };
""",
        "prelu": """
    auto prelu = +[](Scalar& output, Scalar input, Scalar alpha) noexcept {
        output = input > 0 ? input : alpha * input;
    };
""",
        "silu": """
    auto silu = +[](Scalar& output, Scalar input, Scalar alpha) noexcept {
        auto sigmoid = 1 / (1 + std::exp(-input));
        output = input * sigmoid;
    };
""",
        "softmax": """
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
"""
    }

    # normalization functions
    normalization_functions = {
        "LayerNormalization": """
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
""",
        "BatchNormalization": """
template <typename Scalar, int size>
void BatchNormalization(Scalar *outputs, const Scalar *inputs, const Scalar *gamma, const Scalar *beta, const Scalar *mean, const Scalar *variance, const Scalar epsilon) noexcept
{
    for (int i = 0; i < size; ++i)
    {
        outputs[i] = gamma[i] * ((inputs[i] - mean[i]) / std::sqrt(variance[i] + epsilon)) + beta[i];
    }
}
""",
        "BatchNormalization2D": """
template <typename Scalar, int channels, int height, int width>
void BatchNormalization2D(Scalar *outputs, const Scalar *inputs,
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
""",
        "LayerNormalization2D": """
template <typename Scalar, int channels, int height, int width>
void LayerNormalization2D(Scalar *outputs, const Scalar *inputs,
                          const Scalar *gamma, const Scalar *beta,
                          Scalar epsilon) noexcept
{
    for (int c = 0; c < channels; ++c)
    {
        Scalar sum = 0;
        for (int i = 0; i < height * width; ++i)
        {
            int idx = i * channels + c;
            sum += inputs[idx];
        }
        Scalar mean = sum / (height * width);
        Scalar var = 0;
        for (int i = 0; i < height * width; ++i)
        {
            int idx = i * channels + c;
            var += (inputs[idx] - mean) * (inputs[idx] - mean);
        }
        var /= (height * width);
        for (int i = 0; i < height * width; ++i)
        {
            int idx = i * channels + c;
            outputs[idx] = gamma[c] * ((inputs[idx] - mean) / std::sqrt(var + epsilon)) + beta[c];
        }
    }
}
""",
    }

    # convolution functions
    convolution_functions = {
        "Conv1D": """
template <typename Scalar, int out_size>
void Conv1D(Scalar *outputs, const Scalar *inputs, const Scalar *weights, const Scalar *biases,
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
""",
        "Conv1DTranspose": """
template <typename Scalar, int out_channels, int out_length>
void Conv1DTranspose(Scalar *outputs,
                     const Scalar *inputs,
                     const Scalar *weights,
                     const Scalar *biases,
                     int in_channels,
                     int in_length,
                     int kernel_l,
                     int stride_l,
                     int pad_l,
                     activationFunction<Scalar> activation_function,
                     Scalar alpha) noexcept
{
  const int out_size = out_channels * out_length;
  for (int i = 0; i < out_size; ++i) {
    outputs[i] = Scalar(0);
  }
  for (int ic = 0; ic < in_channels; ++ic) {
    for (int il = 0; il < in_length; ++il) {
      const int in_idx = il * in_channels + ic;
      Scalar in_val = inputs[in_idx];

      for (int kl = 0; kl < kernel_l; ++kl) {
        int ol = il * stride_l - pad_l + kl;
        if (ol < 0 || ol >= out_length) continue;

        for (int oc = 0; oc < out_channels; ++oc) {
          int w_idx   = (kl * in_channels + ic) * out_channels + oc;
          int out_idx = ol * out_channels + oc;
          outputs[out_idx] += weights[w_idx] * in_val;
        }
      }
    }
  }
  for (int oc = 0; oc < out_channels; ++oc) {
    for (int ol = 0; ol < out_length; ++ol) {
      int out_idx = ol * out_channels + oc;
      activation_function(
        outputs[out_idx],
        outputs[out_idx] + biases[oc],
        alpha
      );
    }
  }
}
""",
        "Conv2D": """
template <typename Scalar, int out_channels, int out_height, int out_width>
void Conv2D(Scalar *outputs, const Scalar *inputs, const Scalar *weights, const Scalar *biases,
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
""",
        "Conv2DTranspose": """
// Transposed 2D convolution (a.k.a. “deconvolution”)
//   out           : pointer to output buffer of size (H_out * W_out * C_out)
//   in            : pointer to input      buffer of size (H_in  * W_in  * C_in)
//   kernel        : pointer to kernel     buffer of size (k_h    * k_w    * C_out * C_in)
//   bias          : pointer to bias       buffer of size (C_out)  (may be nullptr)
//   in_channels   : number of input  channels (C_in)
//   in_h, in_w    : spatial size of input (H_in, W_in)
//   k_h, k_w      : kernel  spatial dims
//   s_h, s_w      : stride dims
//   p_h, p_w      : padding dims
//   act           : activation        function pointer
//   alpha         : activation “alpha” parameter (e.g. for LeakyReLU)
template<typename Scalar, int C_out, int H_out, int W_out>
void Conv2DTranspose( Scalar*       out,
                      const Scalar* in,
                      const Scalar* kernel,
                      const Scalar* bias,
                      int           in_channels,
                      int           in_h,
                      int           in_w,
                      int           k_h,
                      int           k_w,
                      int           s_h,
                      int           s_w,
                      int           p_h,
                      int           p_w,
                      activationFunction<Scalar> act,
                      Scalar        alpha )
{
    for(int h=0; h < H_out; ++h) {
      for(int w=0; w < W_out; ++w) {
        for(int oc=0; oc < C_out; ++oc) {
          int idx = (h * W_out + w) * C_out + oc;
          out[idx] = bias ? bias[oc] : Scalar(0);
        }
      }
    }

    for(int ic = 0; ic < in_channels; ++ic) {
      for(int ih = 0; ih < in_h; ++ih) {
        for(int iw = 0; iw < in_w; ++iw) {
          Scalar val = in[(ih * in_w + iw) * in_channels + ic];
          int base_h = ih * s_h - p_h;
          int base_w = iw * s_w - p_w;

          for(int kh=0; kh < k_h; ++kh) {
            int oh = base_h + kh;
            if (oh < 0 || oh >= H_out) continue;
            for(int kw=0; kw < k_w; ++kw) {
              int ow = base_w + kw;
              if (ow < 0 || ow >= W_out) continue;

              int out_base = (oh * W_out + ow) * C_out;
              int ker_base = (kh * k_w + kw) * (C_out * in_channels)
                             + ic;  
              for(int oc = 0; oc < C_out; ++oc) {
                int k_idx = ker_base + oc * in_channels;
                out[out_base + oc] += val * kernel[k_idx];
              }
            }
          }
        }
      }
    }

    int total = H_out * W_out * C_out;
    for(int i = 0; i < total; ++i) {
      act(out[i], out[i], alpha);
    }
}
""",
        "Conv3D": """
template <typename Scalar, int out_channels, int out_depth, int out_height, int out_width>
void Conv3D(Scalar *outputs, const Scalar *inputs, const Scalar *weights, const Scalar *biases,
                   int in_channels, int in_depth, int in_height, int in_width,
                   int kernel_d, int kernel_h, int kernel_w, int stride_d, int stride_h, int stride_w,
                   int pad_d, int pad_h, int pad_w,
                   activationFunction<Scalar> activation_function, Scalar alpha) noexcept
{
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
""",
        "Conv3DTranspose": """
template <typename Scalar, int out_channels, int out_depth, int out_height, int out_width>
void Conv3DTranspose(Scalar *outputs, const Scalar *inputs,
                     const Scalar *weights, const Scalar *biases,
                     int in_channels, int in_depth, int in_height, int in_width,
                     int kernel_d, int kernel_h, int kernel_w, int stride_d,
                     int stride_h, int stride_w, int pad_d, int pad_h,
                     int pad_w, activationFunction<Scalar> activation_function,
                     Scalar alpha) noexcept {
  int out_size = out_channels * out_depth * out_height * out_width;
  for (int i = 0; i < out_size; ++i)
    outputs[i] = Scalar(0);
  for (int ic = 0; ic < in_channels; ++ic) {
    for (int id = 0; id < in_depth; ++id) {
      for (int ih = 0; ih < in_height; ++ih) {
        for (int iw = 0; iw < in_width; ++iw) {
          int in_idx = ((id * in_height * in_width) + (ih * in_width) + iw) *
                           in_channels +
                       ic;
          Scalar in_val = inputs[in_idx];
          for (int kd = 0; kd < kernel_d; ++kd) {
            int od = id * stride_d - pad_d + kd;
            if (od < 0 || od >= out_depth)
              continue;
            for (int kh = 0; kh < kernel_h; ++kh) {
              int oh = ih * stride_h - pad_h + kh;
              if (oh < 0 || oh >= out_height)
                continue;
              for (int kw = 0; kw < kernel_w; ++kw) {
                int ow = iw * stride_w - pad_w + kw;
                if (ow < 0 || ow >= out_width)
                  continue;
                for (int oc = 0; oc < out_channels; ++oc) {
                  int w_idx =
                      ((((kd * kernel_h + kh) * kernel_w + kw) * in_channels +
                        ic) *
                           out_channels +
                       oc);
                  int out_idx = (((od * out_height + oh) * out_width) + ow) *
                                    out_channels +
                                oc;
                  outputs[out_idx] += weights[w_idx] * in_val;
                }
              }
            }
          }
        }
      }
    }
  }
  for (int oc = 0; oc < out_channels; ++oc) {
    for (int od = 0; od < out_depth; ++od) {
      for (int oh = 0; oh < out_height; ++oh) {
        for (int ow = 0; ow < out_width; ++ow) {
          int out_idx =
              (((od * out_height + oh) * out_width) + ow) * out_channels + oc;
          activation_function(outputs[out_idx], outputs[out_idx] + biases[oc],
                              alpha);
        }
      }
    }
  }
}
""",
        "DepthwiseConv2D": """
template <typename Scalar>
void DepthwiseConv2D(Scalar *outputs, const Scalar *inputs, const Scalar *weights, const Scalar *biases,
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
""",
        "SeparableConv2D": """

template <typename Scalar>
void DepthwiseForsSeparableConv2D(Scalar *outputs, const Scalar *inputs, const Scalar *weights, const Scalar *biases,
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
void SeparableConv2D(Scalar *outputs, const Scalar *inputs, const Scalar *depthwise_weights, const Scalar *pointwise_weights, const Scalar *biases,
                            int in_channels, int in_height, int in_width,
                            int kernel_h, int kernel_w, int stride_h, int stride_w,
                            int pad_h, int pad_w,
                            activationFunction<Scalar> activation_function, Scalar alpha) noexcept
{
    std::vector<Scalar> depthwise_output(in_height * in_width * in_channels, 0);
    std::vector<Scalar> zero_bias(in_channels, 0);
    DepthwiseForsSeparableConv2D(
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
""",
        "ConvLSTM2DForward": """
template<typename Scalar>
void ConvLSTM2DForward(/* parameters */) noexcept {
    // Stub for ConvLSTM2D.
    // A full implementation would require handling time steps and cell states.
}
""",
    }

    pooling_functions = {
        "MaxPooling2D": """
template <typename Scalar, int pool_height, int pool_width, int stride_h, int stride_w>
void MaxPooling2D(Scalar *outputs, const Scalar *inputs, int in_height, int in_width, int channels) noexcept
{
    int out_height = (in_height - pool_height) / stride_h + 1;
    int out_width = (in_width - pool_width) / stride_w + 1;
    for (int c = 0; c < channels; ++c)
    {
        for (int oh = 0; oh < out_height; ++oh)
        {
            for (int ow = 0; ow < out_width; ++ow)
            {
                Scalar max_val = -std::numeric_limits<Scalar>::infinity();
                for (int ph = 0; ph < pool_height; ++ph)
                {
                    for (int pw = 0; pw < pool_width; ++pw)
                    {
                        int in_h = oh * stride_h + ph;
                        int in_w = ow * stride_w + pw;
                        int idx = (in_h * in_width * channels) + (in_w * channels) + c;
                        if (inputs[idx] > max_val)
                        {
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
""",
        "AvgPooling2D": """
template <typename Scalar, int pool_height, int pool_width, int stride_h, int stride_w>
void AvgPooling2D(Scalar *outputs, const Scalar *inputs, int in_height, int in_width, int channels) noexcept
{
    int out_height = (in_height - pool_height) / stride_h + 1;
    int out_width = (in_width - pool_width) / stride_w + 1;
    for (int c = 0; c < channels; ++c)
    {
        for (int oh = 0; oh < out_height; ++oh)
        {
            for (int ow = 0; ow < out_width; ++ow)
            {
                Scalar sum = 0;
                for (int ph = 0; ph < pool_height; ++ph)
                {
                    for (int pw = 0; pw < pool_width; ++pw)
                    {
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
""",
        "GlobalAvgPooling2D": """
template <typename Scalar>
void GlobalAvgPooling2D(Scalar *output, const Scalar *inputs, int in_height, int in_width, int channels) noexcept
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
""",
    }

    # set every function and append it to cpp_code
    current_activations = set(activation_functions)
    current_activations = {
        ("tanhCustom" if act == "tanh" else act)
        for act in current_activations
        if act is not None and act != "Activation"
    }

    cpp_lambda = """"""

    for act in current_activations:
        if act in lambda_functions:
            cpp_lambda += lambda_functions[act]

    # Deduplicate layer_type list
    unique_layer_types = {lt for lt in layer_type if lt is not None}

    for type in unique_layer_types:
        if type in dense_function:
            cpp_code += dense_function[type]
        if type in normalization_functions:
            cpp_code += normalization_functions[type]
        if type in convolution_functions:
            cpp_code += convolution_functions[type]
        if type in pooling_functions:
            cpp_code += pooling_functions[type]
        # if type == "Conv1D":
        #     cpp_code += convolution_functions["Conv1D"]
        # if type == "Conv2D":
        #     cpp_code += convolution_functions["Conv2D"]
        # if type == "Conv3D":
        #     cpp_code += convolution_functions["Conv3D"]

    return cpp_code, cpp_lambda
