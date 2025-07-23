import tensorflow as tf
from tensorflow.keras.utils import get_custom_objects 
def nonzero_diag_activation(x):     
    # mask = tf.constant(
    #     [1.0 if (i % (M+1) == 0) else 0.0 for i in range(FLAT_DIM)],
    #     dtype=x.dtype
    # )
    # mask = tf.reshape(mask, (1, FLAT_DIM))
    # diag_x = x * (1.0 + tf.exp(-tf.abs(4.0 * x / EPS) + 2.0))
    # return (x * (1.0 - mask)) + (diag_x * mask)
    mask = tf.constant(
    [1.0 if (i % (M+1) == 0) else 0.0 for i in range(FLAT_DIM)],
    dtype=x.dtype
    )
    mask = tf.reshape(mask, (1, FLAT_DIM))
    abs_x = tf.abs(x)
    sign_x = tf.sign(x)
    sign_x = tf.where(tf.equal(sign_x, 0.0), 1.0, sign_x)  
    diag_x = sign_x * tf.maximum(abs_x, EPS)
    return (x * (1.0 - mask)) + (diag_x * mask)
get_custom_objects().update({
    'nonzero_diag_activation': nonzero_diag_activation
})












"""
    auto nonzero_diag_activation = +[](Scalar& output, Scalar input, int index) noexcept
    {
        constexpr int M = 97;
        constexpr int FLAT_DIM = M * M;
        constexpr Scalar EPS = Scalar(1e-16);
        
        bool is_diagonal = (index % (M + 1)) == 0;
        
        if (is_diagonal) {
            Scalar abs_x = std::abs(input);
            Scalar sign_x = (input >= Scalar(0)) ? Scalar(1) : Scalar(-1);
            if (input == Scalar(0)) {
                sign_x = Scalar(1);
            }
            output = sign_x * std::max(abs_x, EPS);
        } else {
            output = input;
        }
    };
"""












"""
    auto nonzero_diag_activation = +[](Scalar& output, Scalar input, int index) noexcept
    {
        constexpr int M = 97;
        constexpr int FLAT_DIM = M * M;
        constexpr Scalar EPS = Scalar(1e-14);  
        
        bool is_diagonal = (index % (M + 1)) == 0;
        
        if (is_diagonal) {
            Scalar abs_term = std::abs(Scalar(4.0) * input / EPS);
            Scalar exp_term = std::exp(-abs_term + Scalar(2.0));
            output = input * (Scalar(1.0) + exp_term);
        } else {
            output = input; 
        }
    };
"""