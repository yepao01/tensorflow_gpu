from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(10)

(x_img_train, y_img_train), (x_img_test, y_img_test) = cifar10.load_data()

# 数据预处理
x_img_train_normalize = x_img_train.astype("float32") / 255.0
x_img_test_normalize = x_img_test.astype("float32") / 255.0
# label
y_img_train_OneHot = np_utils.to_categorical(y_img_train)
y_img_test_OneHot = np_utils.to_categorical(y_img_test, 10)
# 建立线性堆叠模型
model = Sequential()
# 卷积层 池化层 1
model.add(Conv2D(filters=32,
                 kernel_size=(3, 3),
                 input_shape=(32, 32, 3),
                 activation="relu",
                 padding="same"))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))
# 卷积层 池化层 2
model.add(Conv2D(filters=64,
                 kernel_size=(3, 3),
                 activation="relu",
                 padding="same"))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size=(2, 2)))
# 卷积层 池化层 3
model.add(Conv2D(filters=128,
                 kernel_size=(3, 3),
                 activation="relu",
                 padding="same"))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 平坦层
model.add(Flatten())
model.add(Dropout(0.3))
# 隐藏层
model.add(Dense(2500, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(1500, activation="relu"))
model.add(Dropout(0.3))
# 输出层
model.add(Dense(10, activation="softmax"))
print(model.summary())

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])
# 训练
train_history = model.fit(x_img_train_normalize,
                          y_img_train_OneHot,
                          validation_split=0.2,
                          epochs=10,
                          batch_size=200,
                          verbose=1)
# 评估
scores = model.evaluate(x_img_test_normalize,
                        y_img_test_OneHot,
                        verbose=1)
print('scores:', scores[1])
# 预测
prediction = model.predict_classes(x_img_test_normalize)

label_dict = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
              5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}


def plot_images_labels_predictions(images, labels, prediction, idx, num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num > 25:
        num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1 + i)
        ax.imshow(images[idx], cmap="binary")
        title = str(i) + "," + label_dict[labels[i][0]]
        if len(prediction) > 0:
            title += "=>" + label_dict[prediction[i]]
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx += 1
    plt.show()


plot_images_labels_predictions(x_img_test,
                               y_img_test,
                               prediction,
                               0, 10)
