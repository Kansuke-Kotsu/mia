'''basic library'''
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.models import load_model
import matplotlib.pyplot as plt
import time

'''config'''
id = 0
rate = 1
epoch = 100
model_params = "model/model_0050.h5"
cal_grad = tf.keras.losses.CategoricalCrossentropy()

def main():
    model = load_model(model_params)
    # vector ==> image
    z = tf.random.normal((1, 28, 28), 0, 1, tf.float32, seed=1)

    # mia
    for i in range(epoch):
        y = np.zeros(9, dtype=np.float32)
        y = np.insert(y, id, 1)
        y = tf.multiply(y, 1)

        plt.imshow(z[0])
        plt.savefig('result/image_{:04d}.png'.format(i))
        # 元の入力テンソル x に対する z の微分
        with tf.GradientTape() as t:
            t.watch(z)
            x = model(z)[0]
            print(x[0])
            loss = cal_grad(y, x)
        grad = t.gradient(loss, z)
        z -= rate * grad






if __name__ == '__main__':
    main()

    