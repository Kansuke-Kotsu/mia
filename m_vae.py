# coding: UTF-8

import os
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"

import sys
import csv
import cv2
import argparse
import numpy as np
import chainer
from chainer import cuda
from chainer import Variable
from chainer import serializers
from data_io import read_csv
from data_io import read_image_in_chainer_format
from network import myCNN
from VAE import myVAE


# コマンドライン引数のパース
parser = argparse.ArgumentParser(description = 'MIA against CNN-based face recognizer')
parser.add_argument('--gpu', '-g', default=0, type=int, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--p_id', '-i', default=0, type=int, help='PARSON ID ')
parser.add_argument('--iter', '-e', default=30, type=int, help='number of iteration')
parser.add_argument('--number', '-n', default=10, type=int, help='number of images (generation image)')
parser.add_argument('--mode', '-m', default=0, type=int, help='running mode')
args = parser.parse_args()

# コンフィグ
INITIAL_IMAGE = './average.png'
INTERVAL = 1000
PERSON_ID = args.p_id
WINDOW_NAME = 'Generated'
MODEL_ATTACK_FILE = '../DataSet/VggFace2/Attack/model_5.model'
MODEL_VALID_FILE = '../DataSet/VggFace2/Valid/model_10.model'
VAE_MODEL_FILE = '../DataSet/VggFace2/DeepGenerativeModel/models_VAE/VAE_20.model'
generationImages = args.number




#    arr: 二次元または三次元の 32bit-float numpy array
#        （二次元の場合は shape=[height, width]，三次元の場合は shape=[n_channels, height, width] ）
def show_as_image(window_name, arr, interval, save=False,file_name='./results/mia_result.png'):

    d = len(arr.shape)
    img = (arr * 255).astype(np.uint8)

    if d == 2:
        cv2.imshow(window_name, img)
        if save == True:
            cv2.imwrite(file_name, img)
    elif d == 3:
        img2 = img.transpose(1, 2, 0)
        cv2.imshow(window_name, img2)
        if save == True:
            cv2.imwrite(file_name, img2)
    cv2.waitKey(interval)


# 実際に実行する処理
if __name__ == '__main__':

    #n_classes = 60 # 攻撃対象の顔認識器におけるクラス数（手動設定）
    n_classes = 2141
    # コマンドライン引数により指定されたパラメータを変数に格納し，標準出力に表示
    GPU_ID = args.gpu # 使用するGPUのID
    N_ITER = args.iter # 反復回数
    print('GPU ID: {0}'.format(GPU_ID), file=sys.stderr)
    print('{0}番目の人の顔を生成'.format(PERSON_ID), file=sys.stderr)
    #print('更新率 : {0}'.format(UPDATING_RATE), file=sys.stderr)
    print('Num. of iterations: {0}'.format(N_ITER), file=sys.stderr)
    print('', file = sys.stderr)
    

    # CNNモデルの準備
    model_at = myCNN(n_classes) # 攻撃対象モデルを作成
    model_ev = myCNN(n_classes+4) # 評価用モデルを作成
    serializers.load_npz(MODEL_ATTACK_FILE, model_at) # 学習済みのネットワークパラメータをロード
    serializers.load_npz(MODEL_VALID_FILE, model_ev) # 学習済みのネットワークパラメータをロード
    vae_model = myVAE(128, 128)
    serializers.load_npz(VAE_MODEL_FILE, vae_model)
    if GPU_ID >= 0:
        cuda.get_device_from_id(GPU_ID).use()
        model_at.to_gpu() # モデルを GPU に載せる
        model_ev.to_gpu()
        vae_model.to_gpu()
        use_cpu = False
    else:
        use_cpu = True
    xp = np if GPU_ID < 0 else cuda.cupy # xp: CPUモードのときは numpy，GPUモードのときは cupy の意味になる

    # 学習処理ループ
    cv2.namedWindow(WINDOW_NAME)
    chainer.config.train = False
         

    if args.mode == 0: 
        # 画像を1枚生成
        # 更新ごとに経過を表示
        with chainer.using_config('train', False):
            z = xp.random.rand(1, 1024)
            z = xp.array(z,dtype=xp.float32)
            z = Variable(z)
            show_as_image(WINDOW_NAME, chainer.cuda.to_cpu(vae_model.decode(z).data)[0],INTERVAL,save=True,file_name='./results/result_0.png')
        #print(z)
        for it in range(N_ITER):
            # 勾配を求める
            model_at.cleargrads()
            model_ev.cleargrads()
            with chainer.using_config('train', False):
                x = vae_model.decode(z)      # ベクトル → 画像 
                g, loss, score, rate = model_at.get_grad(x, PERSON_ID, use_cpu) # 攻撃用での各値（勾配，誤差，スコア，認識順位）
                g2, loss2, score2, rate2 = model_ev.get_grad(x, PERSON_ID, use_cpu) # 評価用での各値（勾配，誤差，スコア，認識順位）
                x.grad = xp.asarray(g)
                x.backward(retain_grad=True)
                g = z.grad
            # 入力値を更新
            g = Variable(g)
            UPDATING_RATE = 0.1
            z -= UPDATING_RATE * g
            print('score({0})'.format(it+1), file=sys.stderr)
            print('Attack / Valid : {0}    /   {1}'.format(score, score2), file=sys.stderr)
            print('認識率第 {0} 位'.format(rate2), file=sys.stderr) # 評価用での順位
            show_as_image(WINDOW_NAME, chainer.cuda.to_cpu(vae_model.decode(z).data)[0],INTERVAL,save=True,file_name='./results/result_{0}_{1}_{2}.png'.format(it+1, score, score2))
            if score >= 0.99:
                break


    if args.mode == 1:
        # 画像を n枚生成
        # 生成結果のみを n枚保存
        for GenerationImages in range (generationImages):
            print('{0}回目:'.format(GenerationImages+1), file=sys.stderr)
            with chainer.using_config('train', False):
                z = xp.random.rand(1, 1024)
                z = xp.array(z,dtype=xp.float32)
                z = Variable(z)
            #print(z)
            show_as_image(WINDOW_NAME, chainer.cuda.to_cpu(vae_model.decode(z).data)[0],INTERVAL,save=False)

            for it in range(N_ITER):
                # 勾配を求める
                model_at.cleargrads()
                with chainer.using_config('train', False):
                    x = vae_model.decode(z)       # ベクトル → 画像 
                    g, loss, score = model_at.get_grad(x, PERSON_ID, use_cpu)
                    x.grad = xp.asarray(g)
                    x.backward(retain_grad=True)
                    g = z.grad
                # 入力値を更新
                g = Variable(g)
                UPDATING_RATE = 0.1
                z -= UPDATING_RATE * g
                print("{0}:Score {1}".format(it+1,score))
                if score >= 0.9:
                    break
            show_as_image(WINDOW_NAME, chainer.cuda.to_cpu(vae_model.decode(z).data)[0],INTERVAL,save=True,file_name='./results/result_{0}_{1}_{2}.png'.format(GenerationImages, it+1, score))

   

    # モデルをCPUメモリ上に戻しておく
    if GPU_ID >= 0:
        model_at.to_cpu()
        model_ev.to_cpu()
        vae_model.to_cpu()