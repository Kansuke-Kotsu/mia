import os
import pathlib

import numpy as np
import pandas as pd 
import tensorflow as tf

import glob
import cv2
import matplotlib
import matplotlib.pyplot as plt

CSV_FILE_PATH = "trainlog.csv"
if not os.path.exists(CSV_FILE_PATH): 
    pathlib.Path(CSV_FILE_PATH).touch()

# Load MNIST dataset from tensorflow
mnist = tf.keras.datasets.mnist
(X_train, y_train),(X_test, y_test) = mnist.load_data()
del mnist

print("X_train : ", X_train.shape)
print("y_train : ", y_train.shape)
print("X_test : ", X_test.shape)
print("y_test : ", y_test.shape)

'''for i in [1,10,100]:
    print("y_train", "(i="+str(i)+"): ", y_train[i])
    print("X_train", "(i="+str(i)+"): ")    
    plt.imshow(X_train[i], cmap='gray')
    plt.show()'''

print("X_train min", X_train.min())
print("X_train max", X_train.max())
# Min-Max Normalization
X_train, X_test = X_train/255.0, X_test/255.0
print("X_train min", X_train.min())
print("X_train max", X_train.max())

# モデル
model = tf.keras.models.Sequential([
    # (None, 28, 28) -> (None, 784)
    tf.keras.layers.Flatten(input_shape=(28, 28), name='input'),
    # Layer1: Linear mapping: (None, 784) -> (None, 512)
    tf.keras.layers.Dense(512, name='fc_1'),
    # Activation function: ReLU
    tf.keras.layers.Activation(tf.nn.relu, name='relu_1'),
    # Layer2: Linear mapping: (None, 512) -> (None, 256)
    tf.keras.layers.Dense(256, name='fc_2'),
    # Activation function: ReLU
    tf.keras.layers.Activation(tf.nn.relu, name='relu_2'),
    # Layer3: Linear mapping: (None, 256) -> (None, 256)
    tf.keras.layers.Dense(256, name='fc_3'),
    # Activation function: ReLU
    tf.keras.layers.Activation(tf.nn.relu, name='relu_3'),
    # Layer4: Linear mapping: (None, 256) -> (None, 10)
    tf.keras.layers.Dense(10, name='dense_3'),
    # Activation function: Softmax
    tf.keras.layers.Activation(tf.nn.softmax, name='softmax')
])

# View model architecture
model.summary()

# Compiling
# Set model & training information into machine memory (CPU or GPU)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Set callback functions which are called during model training
callbacks = []
callbacks.append(tf.keras.callbacks.CSVLogger(CSV_FILE_PATH))

# Train model
history = model.fit(X_train, y_train, 
                    batch_size=100, 
                    epochs=1,
                    verbose=1, 
                    validation_data=(X_test, y_test),
                    callbacks=callbacks)

# Model evaluation
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=1)
print("loss(train): {:.4}".format(train_loss))
print("accuracy(train): {:.4}".format(train_acc))

print()

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
print("loss(test): {:.4}".format(test_loss))
print("accuracy(test): {:.4}".format(test_acc))

for i in [0,1,2]:
    y_true = y_test[i]
    y_pred = model.predict(X_test[i].reshape(1,28,28))[0]
    #y_pred = model.predict_classes(X_test[i].reshape(1,28,28))[0]
    print("y_test_pred", "(i="+str(i)+"): ", y_pred)
    print("y_test_true", "(i="+str(i)+"): ", y_true)
    print("X_test", "(i="+str(i)+"): ")    
    plt.imshow(X_test[i], cmap='gray')
    plt.show()


fig = plt.figure(figsize=(12, 8))

ROW = 4
COLUMN = 5

for i in range(ROW * COLUMN):
    y_true = y_test[i]
    y_pred = model.predict(X_test[i].reshape(1,28,28))[0]
    #y_pred = model.predict_classes(X_test[i].reshape(1,28,28))[0]
    if y_true == y_pred:
        result = "True" # Correct answer from the model
    else:
        result = "False" # Incorrect answer from the model
    
    plt.subplot(ROW, COLUMN, i+1)
    plt.imshow(X_test[i], cmap='gray')
    plt.title("No.{} - {}\ny_true:{}, y_pred:{}".format(i, result, y_true, y_pred))
    plt.axis("off")

fig.tight_layout()
fig.show()

# 全ての画像ファイルのパスを取得する
files = glob.glob("images/test_7.png")
# パスの画像ファイルを読み込み
image = cv2.imread(files[0],cv2.IMREAD_GRAYSCALE)
# resize
image = cv2.resize(image, (28, 28))
result = model.predict(image.reshape(1,28,28))[0]
#result = model.predict_classes(image.reshape(1,28,28))[0]
print("actual : 7")
print("predict: ", result)
# save model as keras instance
#ins_path = 'mnist_model_v0.h5'
model.save("mnist_model")