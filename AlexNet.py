from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
import numpy as np
import os
import cv2
import time


resize = 224
def load_data():
    imgs = os.listdir("./train/")
    num = len(imgs)
    train_data = np.empty((2000,resize,resize,3),dtype="int32")
    train_label = np.empty((2000,),dtype="int32")
    test_data = np.empty((2000, resize, resize, 3), dtype="int32")
    test_label = np.empty((2000, ), dtype="int32")
    for i in range(3000,5000):
        if i%2:
            train_data[i-3000] = cv2.resize(cv2.imread("./train/" + "dog." + str(i) + ".jpg"),(resize,resize))
            train_label[i-3000] = 1
        else:
            train_data[i-3000] = cv2.resize(cv2.imread("./train/" + "cat." + str(i) + ".jpg"),(resize,resize))
            train_label[i-3000] = 0
    for i in range(5000, 7000):
        if i % 2:
            test_data[i - 5000] = cv2.resize(cv2.imread('./train/' + 'dog.' + str(i) + '.jpg'), (resize, resize))
            test_label[i - 5000] = 1
        else:
            test_data[i - 5000] = cv2.resize(cv2.imread('./train/' + 'cat.' + str(i) + '.jpg'), (resize, resize))
            test_label[i - 5000] = 0
    return train_data, train_label, test_data, test_label

train_data, train_label, test_data, test_label = load_data()
print(len(train_data))
train_data, test_data = train_data.astype('float32'), test_data.astype('float32')
train_data, test_data = train_data/255, test_data/255

train_label = to_categorical(train_label, 2)
test_label = to_categorical(test_label, 2)


#5个卷积层  3个全连接层
#96个卷积核 卷积核大小（11,11）  步长（4,4） 传入格式（227,227,3）填补='valid'
# AlexNet
model = Sequential()
# 第一段 卷积 归一化 池化
model.add(Conv2D(filters=96, kernel_size=(11, 11),
                 strides=(4, 4), padding='valid',
                 input_shape=(resize, resize, 3),
                 activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3),
                       strides=(2, 2),
                       padding='valid'))
# 第二段 卷积 归一化 池化
model.add(Conv2D(filters=256, kernel_size=(5, 5),
                 strides=(1, 1), padding='same',
                 activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3),
                       strides=(2, 2),
                       padding='valid'))
# 第三段 卷积*3 池化
model.add(Conv2D(filters=384, kernel_size=(3, 3),
                 strides=(1, 1), padding='same',
                 activation='relu'))
model.add(Conv2D(filters=384, kernel_size=(3, 3),
                 strides=(1, 1), padding='same',
                 activation='relu'))
model.add(Conv2D(filters=256, kernel_size=(3, 3),
                 strides=(1, 1), padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3),
                       strides=(2, 2), padding='valid'))
# 第四段
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.5))

# Output Layer
model.add(Dense(2,activation='sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.summary()
while 1:
    try:
        model.load_weights("./SaveModel/AlexNet.h5")
        print("加载模型成功。")
    except:
        print("加载失败，开始训练一个新模型。")

    model.fit(train_data,train_label,
              batch_size=32,
              epochs=10,
              validation_split=0.2,
              shuffle=True,
              verbose=2)
    model.save_weights("./SaveModel/AlexNet.h5")
    print("保存模型")
    time.sleep(300)

# print(model.evaluate(train_data,train_label,verbose=2))
# print(model.evaluate(test_data,test_label,verbose=2))
