from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.utils import to_categorical
import mnist

def cnnModel(X_train, y_train, X_Test, y_test):
    num_filters = 8
    filter_size = 3
    pool_size = 2

    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, strides=1,
                     activation='relu',
                     input_shape=()))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(
        loss=categorical_crossentropy,
        optimizer=Adam,
        metrics=['accuracy']
    )

    model.fit(X_train, to_categorical(y_train), epochs=5)

    model.evaluate(X_Test, to_categorical(y_test))

train_images = mnist.train_images()
train_labels = mnist.train_labels()

test_images = mnist.test_images()
test_labels = mnist.test_labels()

cnnModel(train_images, train_labels, test_images, test_labels)