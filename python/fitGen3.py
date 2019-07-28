from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution1D, Convolution2D, MaxPooling1D
from keras.utils import np_utils


def myGenerator():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    y_train = np_utils.to_categorical(y_train,10)
    X_train = X_train.reshape(X_train.shape[0], img_rows* img_cols,1)
    X_test = X_test.reshape(X_test.shape[0], img_rows* img_cols,1)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    while 1:
        for i in range(1875): # 1875 * 32 = 60000 -> # of training samples
            if i%125==0:
                print "i = " + str(i)
            yield X_train[i*32:(i+1)*32], y_train[i*32:(i+1)*32]

batch_size = 128
nb_classes = 10
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3
model = Sequential()

model.add(Convolution1D(32, 1, activation='relu',input_shape=(28*28,1)))
model.add(MaxPooling1D(pool_size=(2)))
model.add(Dropout(0))
 
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta')

#model.fit_generator(myGenerator(), steps_per_epoch = 500 , epochs = 10, verbose=2 )
