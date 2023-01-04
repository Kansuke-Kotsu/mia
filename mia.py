'''basic library'''
import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
import keras.backend as K

'''config'''
id = 0
rate = - 0.01
epoch = 50
model = load_model('model/')

def cal_grad(model, x, y):
    class_output = model.output[:,5]  # Tensor / クラススコア
    grad_tensor = K.gradients(class_output, model.input)[0]  # Tensor / クラススコアに対する入力の勾配
    grad_func = K.function([model.input], [grad_tensor])  #  勾配の値を算出するための関数
    x= np.expand_dims(train_images[0], axis=0)  #画像自体は大きさが(28,28) なので(1,28,28)にする
    grad = grad_func([x])[0][0]  # ndarray: 算出された勾配の値
    loss = tf.keras.losses.binary_crossentropy(x, y)
    return class_output, loss, grad

def main():
    # vector ==> image
    z = tf.random.normal([28, 28], 0, 1, tf.float32, seed=1)
    #print(z)
    #print(z.shape)

    # mia
    for i in range(epoch):
        y = id * np.ones()
        x = model.predict(z)
        # 勾配計算
        class_output, loss, grad = cal_grad(model=model, x=x, y=y)
        #print(grad)
        # 逆伝播
        z -= rate * grad
        print('{0}回目: ', format(i+1))
        print('score: {0}', format(class_output))


main()

    