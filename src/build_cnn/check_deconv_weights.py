#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

def flat_index(kh, kw, oc, ic, kernel_h, kernel_w, out_ch, in_ch):
    """
    Given tensor shape [kh][kw][out_ch][in_ch],
    returns the row-major flat index of element (kh,kw,oc,ic).
    """
    return ((kh * kernel_w + kw) * out_ch + oc) * in_ch + ic

def main():
    # 1) Load your model
    model = load_model("cnn6.h5")
    
    # 2) Find the Conv2DTranspose layer
    deconv = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2DTranspose):
            deconv = layer
            break
    if deconv is None:
        raise RuntimeError("No Conv2DTranspose layer found in the model!")
    
    # 3) Extract kernel weights
    #    weights.shape == (kernel_h, kernel_w, out_ch, in_ch)
    weights, biases = deconv.get_weights()
    kh, kw, out_ch, in_ch = weights.shape
    flat = weights.flatten()  # C-order
    
    print(f"Conv2DTranspose kernel shape: {weights.shape}")
    print(f"Flattened length: {flat.shape[0]}\n")
    
    # 4) Define some sample indices to check
    samples = [
        (0, 0, 0, 0),
        (1, 1, 5, 10),
        (2, 2, 31, 31),
        (0, 2, 16, 7),
        (2, 0, 8, 24)
    ]
    
    # 5) Compare direct vs. flattened
    for (ch, cw, oc, ic) in samples:
        val_direct = weights[ch, cw, oc, ic]
        idx = flat_index(ch, cw, oc, ic, kh, kw, out_ch, in_ch)
        val_flattened = flat[idx]
        print(f"weights[{ch},{cw},{oc},{ic}] = {val_direct:.9e}")
        print(f"flat[{idx}]           = {val_flattened:.9e}")
        print("  → match?" , np.isclose(val_direct, val_flattened))
        print()
    
    # 6) If any of these print False, you know your flatten order/Axes are off.

if __name__ == "__main__":
    main()
