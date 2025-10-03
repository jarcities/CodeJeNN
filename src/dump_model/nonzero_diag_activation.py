from tensorflow.keras.utils import register_keras_serializable
import tensorflow as tf


@register_keras_serializable()
def nonzero_diag_activation(x):
    # eps = 1e-4
    # mask = tf.constant(diag_mask_np, dtype=x.dtype)
    # mask = tf.reshape(mask, (1, OUTPUT_DIM))
    # diag_elements = x * mask
    # non_diag_elements = x * (1.0 - mask)
    # tanh_scaled = tf.tanh(diag_elements)
    # sign_tanh = tf.sign(tanh_scaled)
    # sign_tanh = tf.where(tf.equal(sign_tanh, 0), tf.ones_like(sign_tanh), sign_tanh)
    # abs_tanh = tf.abs(tanh_scaled)
    # eps_tensor = tf.fill(tf.shape(abs_tanh), tf.cast(eps, x.dtype))
    # one_tensor = tf.fill(tf.shape(abs_tanh), tf.cast(1.0, x.dtype))
    # scaled_diag = sign_tanh * (abs_tanh * (one_tensor - eps_tensor) + eps_tensor)
    # return non_diag_elements + scaled_diag * mask
    return 0


"""
    auto nonzero_diag_activation = [](Scalar& output, Scalar input, int index) noexcept
    {
        constexpr Scalar EPS = Scalar(1e-4);
        constexpr Scalar ONE = Scalar(1.0);
        int r = get_lu_perm_row_index(index);
        int c = get_lu_perm_col_index(index);
        
        if (r == c) {
            Scalar tanh_val = std::tanh(input);
            Scalar sign_val = (tanh_val >= Scalar(0)) ? ONE : Scalar(-1);
            if (tanh_val == Scalar(0)) sign_val = ONE;
            Scalar abs_tanh = std::abs(tanh_val);
            
            // Formula: sign * (|tanh| * (1 - eps) + eps)
            output = sign_val * (abs_tanh * (ONE - EPS) + EPS);
        }
        else {
            output = input;
        }
    };
"""