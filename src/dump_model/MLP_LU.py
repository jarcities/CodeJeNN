import tensorflow as tf
from tensorflow.keras.utils import get_custom_objects 
def nonzero_diag(x):     
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
    'nonzero_diag': nonzero_diag
})












"""
    auto nonzero_diag = +[](Scalar& output, Scalar input, int index) noexcept
    {
        constexpr int M = 97;
        constexpr int FLAT_DIM = M * M;
        constexpr Scalar EPS = Scalar(1e-4);
        
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
    auto nonzero_diag = +[](Scalar& output, Scalar input, int index) noexcept
    {
        constexpr int M = 202;  // Based on your Python code
        constexpr int OUTPUT_DIM = /* your actual output dimension */;
        constexpr Scalar EPS = Scalar(1e-4);
        
        // Create the diagonal mask mapping based on mask_out sparsity pattern
        // This would need to be precomputed and stored as a constant array
        // For now, assuming you have a way to determine if index corresponds to diagonal
        static constexpr bool diag_mask[OUTPUT_DIM] = {
            // This array should be precomputed from your mask_out sparsity pattern
            // where diag_mask[i] = true if the i-th non-zero element corresponds to a diagonal
        };
        
        if (index < OUTPUT_DIM && diag_mask[index]) {
            // This is a diagonal element - apply the nonzero constraint
            Scalar abs_x = std::abs(input);
            Scalar sign_x = (input >= Scalar(0)) ? Scalar(1) : Scalar(-1);
            if (input == Scalar(0)) {
                sign_x = Scalar(1);
            }
            output = sign_x * std::max(abs_x, EPS);
        } else {
            // Non-diagonal element - pass through unchanged
            output = input;
        }
    };
"""

