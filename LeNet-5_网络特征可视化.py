#----------------------------------各个层特征可视化-------------------------------
#查看输入图片
from keras import backend as K
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
#加载前面保存的模型
model=load_model(r'D:/test/tensorflow_gpu/LeNet-5_model.h5')
#查看输入图片
fig1,ax1 = plt.subplots(figsize=(4,4))
ax1.imshow(np.reshape(x_test[12], (28, 28)))
plt.show()

image_arr=np.reshape(x_test[12], (-1,28, 28,1))
#可视化第一个MaxPooling2D
layer_1 = K.function([model.layers[0].input], [model.layers[1].output])
# 只修改inpu_image
f1 = layer_1([image_arr])[0]
# 第一层卷积后的特征图展示，输出是（1,12,12,6），（样本个数，特征图尺寸长，特征图尺寸宽，特征图个数）
re = np.transpose(f1, (0,3,1,2))
for i in range(6):
    plt.subplot(2,4,i+1)
    plt.imshow(re[0][i]) #,cmap='gray'
plt.show()
#可视化第二个MaxPooling2D
layer_2 = K.function([model.layers[0].input], [model.layers[3].output])
f2 = layer_2([image_arr])[0]
# 第一层卷积后的特征图展示，输出是（1,4,4,16），（样本个数，特征图尺寸长，特征图尺寸宽，特征图个数）
re = np.transpose(f2, (0,3,1,2))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.imshow(re[0][i]) #, cmap='gray'
plt.show()