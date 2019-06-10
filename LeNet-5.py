from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((-1,28,28,1))
y_train = to_categorical(y_train,10)
x_test = x_test.reshape((-1,28,28,1))
y_test = to_categorical(y_test,10)

#model=load_model('E:/LeNet/LeNet-5_model.h5')
model = Sequential()
model.add(Conv2D(6,(5,5),strides=(1,1),input_shape=(28,28,1),padding='valid',activation='relu',kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(16,(5,5),strides=(1,1),padding='valid',activation='relu',kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(120,activation='relu'))
model.add(Dense(84,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

# model.fit(x_train,y_train,batch_size=100,epochs=10,shuffle=True)
# model.save(r'D:/test/tensorflow_gpu/LeNet-5_model.h5')
#[0.10342620456655367 0.9834000068902969]
# loss, accuracy=model.evaluate(x_test, y_test)
# print(loss, accuracy)