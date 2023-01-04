import os
import sys
import numpy as np

import tensorflow as tf

id = 1
cross_entropy = tf.keras.losses.binary_crossentropy(from_logits=True)
y = id * np.ones(, dtype=np.int32)

def get_grad(self, x, id):
    y = id * np.ones(x.data.shape[0], dtype=np.int32)
    z = cross_entropy(y, x)



