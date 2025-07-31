import tensorflow as tf
from tensorflow.keras.utils import get_custom_objects
indices = np.where(mask_out)[0]
diag_flat = np.arange(M) * (M + 1)
diag_mask_np = np.where(np.in1d(indices, diag_flat), 1.0, 0.0)
@tf.function
def nonzero_diag(x):
    eps = 1e-4
    mask = tf.constant(diag_mask_np, dtype=x.dtype)
    mask = tf.reshape(mask, (1, OUTPUT_DIM))
    sign_x = tf.sign(x)
    sign_x = tf.where(tf.equal(sign_x, 0), tf.ones_like(sign_x), sign_x)
    abs_x = tf.abs(x)
    eps_t = tf.fill(tf.shape(abs_x), tf.cast(eps, x.dtype))
    diag_x = sign_x * tf.maximum(abs_x, eps_t)
    return x * (1.0 - mask) + diag_x * mask
get_custom_objects().update({'nonzero_diag': nonzero_diag})












"""
    auto nonzero_diag = +[](Scalar& output, Scalar input, int index) noexcept
    {
        constexpr Scalar EPS = Scalar(1e-4);

        // ask “which row/col does this output slot correspond to?”
        int r = get_lu_perm_row_index(index);
        int c = get_lu_perm_col_index(index);

        if (r == c) {
            // clamp to ±EPS
            Scalar abs_x  = std::abs(input);
            Scalar sign_x = (input >= Scalar(0) ? Scalar(1) : Scalar(-1));
            if (input == Scalar(0)) sign_x = Scalar(1);
            output = sign_x * std::max(abs_x, EPS);
        }
        else {
            output = input;
        }
    };
"""