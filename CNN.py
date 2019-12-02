
from click._compat import raw_input
from keras.models import Sequential
from keras.layers import Conv3D, MaxPooling3D, Dense, Flatten
from keras.losses import categorical_crossentropy
from keras.utils import to_categorical
from sklearn.model_selection import KFold
from keras.datasets import mnist



from preprocess import Preprocess


def cnnModel(X_train, y_train, X_Test, y_test):
    num_filters = 8
    filter_size = 3
    pool_size = 2

    model = Sequential()

    model.add(Conv3D(32, kernel_size=3, strides=1,
                     activation='relu',
                     padding='valid',
                     input_shape=(5, 48, 48, 1)))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=1))
    model.add(Conv3D(64, (3, 3, 3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(1, 3, 3)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(
        loss=categorical_crossentropy,
        optimizer='adam',
        metrics=['accuracy']
    )

    model.fit(X_train, y_train, epochs=30)

    score = model.evaluate(X_Test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
# img_x, img_y = 28, 28
# num_classes = 10
#
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
# x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)
# input_shape = (img_x, img_y, 1)
#
# # convert the data to the right type
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')
#
# # convert class vectors to binary class matrices - this is for use in the
# # categorical_crossentropy loss below
# y_train = to_categorical(y_train, 10)
# y_test = to_categorical(y_test, 10)

preprocess = Preprocess()
X, y = preprocess.get_all_data()
kf = KFold(n_splits=10, shuffle=True)
for train_index, test_index in kf.split(X):
    train_X, train_y = X[train_index], y[train_index]
    test_X, test_y = X[test_index], y[test_index]

    train_X = train_X.reshape(train_X.shape[0], 5, 48, 48, 1)
    test_X = test_X.reshape(test_X.shape[0], 5, 48, 48, 1)
    train_y = train_y.reshape(train_y.shape[0])
    train_y = to_categorical(train_y, 3)
    test_y = test_y.reshape(test_y.shape[0])
    test_y = to_categorical(test_y, 3)

    cnnModel(train_X, train_y, test_X, test_y)

# X_1 = X[0:4000]
# y_1 = y[0:4000]
# X_2 = X[4000:X.shape[0]]
# y_2 = y[4000:y.shape[0]]
#
# X_1 = X_1.reshape(X_1.shape[0], 48, 48, 1)
# X_2 = X_2.reshape(X_2.shape[0], 48, 48, 1)
# y_1 = y_1.reshape(y_1.shape[0])
# y_1 = to_categorical(y_1, 3)
# y_2 = y_2.reshape(y_2.shape[0])
# y_2 = to_categorical(y_2, 3)
#
# cnnModel(X_1, y_1, X_2, y_2)




