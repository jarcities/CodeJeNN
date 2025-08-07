import tensorflow as tf
from tensorflow.keras.utils import get_custom_objects
import numpy as np
import tensorflow.keras.backend as K

# Define your constants here
OUTPUT_DIM = 7652  # Set this to your actual output dimension
M = int(np.sqrt(OUTPUT_DIM))  # Assuming square matrix
mask_out = np.ones(OUTPUT_DIM, dtype=int)  # Set appropriate mask
full_size       = M * M
all_flat_idx    = np.arange(full_size)
output_flat_idx = all_flat_idx[mask_out]
diag_flat_idx   = np.arange(M) * (M + 1)
diag_mask_np    = np.isin(output_flat_idx, diag_flat_idx).astype(np.float64)
diag_mask       = tf.constant(diag_mask_np, dtype=tf.float64)
def nonzero_diag(x):
    eps  = K.epsilon() #1e-7
    print("epsilon =", K.epsilon())
    sign = tf.sign(x)
    sign = tf.where(tf.equal(sign, 0), tf.ones_like(sign), sign)
    return x + eps * sign * diag_mask
get_custom_objects().update({"nonzero_diag": nonzero_diag})












"""
    auto nonzero_diag = +[](Scalar& output, Scalar input, int index) noexcept
    {
        constexpr Scalar EPS = Scalar(1e-4);

        int r = get_lu_perm_row_index(index);
        int c = get_lu_perm_col_index(index);

        if (r == c) {
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