'''basic library'''
import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
'''for VAE'''
from vae import compute_loss
'''for GAN'''
from gan import generator_loss

'''config'''
# define recognition
rec_model = load_model('model/')
# define generator
vae_model = load_model('model/')
gan_model = load_model('model/')
dif_model = load_model('model/')


def main():
    # vector ==> image
    z = tf.random.normal(shape=(100, 2))
    vae_image = vae_model.decode(z)
    gan_image = gan_model.generator(z, training=False)

    # mia
    # 勾配計算
    grad, loss, score = rec_model.get_grad(x, PERSON_ID, use_cpu)
    g_vae = x_vae.grad
    g_gan = x_gan.grad
    # 逆伝播
    z -= rate * g_vae
    z -= rate * g_gan
    