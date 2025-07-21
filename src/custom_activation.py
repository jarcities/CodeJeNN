import tensorflow as tf

def nonzero_diag_activation(x):
    # ---- adjust these to match your globals ----
    M = 97                    # your matrix size
    epsilon = 1e-6            # how far from zero you want to push
    # --------------------------------------------
    FLAT_DIM = M * M

    # build a constant mask vector of shape (FLAT_DIM,)
    # mask[i] = 1.0 iff i is a diagonal index (i % (M+1) == 0)
    mask = tf.constant(
        [1.0 if (i % (M+1) == 0) else 0.0 for i in range(FLAT_DIM)],
        dtype=x.dtype
    )
    # reshape to (1,FLAT_DIM) so it broadcasts over the batch
    mask = tf.reshape(mask, (1, FLAT_DIM))

    # compute sign(x), but treat zero as +1
    sign = tf.sign(x)
    sign = tf.where(tf.equal(sign, 0), tf.ones_like(sign), sign)

    # only on the diagonal entries, add epsilon in the sign direction
    return x + mask * sign * epsilon

from tensorflow.keras.utils import get_custom_objects
get_custom_objects().update({
    'nonzero_diag_activation': nonzero_diag_activation
})