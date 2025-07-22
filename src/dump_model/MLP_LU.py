import tensorflow as tf

def nonzero_diag_activation(x):
    M = 97                   
    epsilon = 1e-6           
    FLAT_DIM = M * M
    mask = tf.constant(
        [1.0 if (i % (M+1) == 0) else 0.0 for i in range(FLAT_DIM)],
        dtype=x.dtype
    )
    mask = tf.reshape(mask, (1, FLAT_DIM))
    diag_x = x * (1.0 + tf.exp(-tf.abs(4.0 * x / epsilon) + 2.0))
    return (x * (1.0 - mask)) + (diag_x * mask)

from tensorflow.keras.utils import get_custom_objects
get_custom_objects().update({
    'nonzero_diag_activation': nonzero_diag_activation
})












"""
    auto nonzero_diag_activation = +[](Scalar& output, Scalar input, Scalar alpha) noexcept
    {
        constexpr int M = 97;
        constexpr Scalar epsilon = Scalar(1e-6);
        int index = static_cast<int>(alpha);
        bool is_diagonal = (index % (M + 1)) == 0;
        if (is_diagonal) {
            output = input * (Scalar(1) + std::exp(-std::abs(Scalar(4) * input / epsilon) + Scalar(2)));
        } else {
            output = input;
        }
    };
"""