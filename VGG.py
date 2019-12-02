from click._compat import raw_input
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.losses import categorical_crossentropy
from keras.utils import to_categorical
from sklearn.model_selection import KFold
from keras.datasets import mnist



from preprocess import Preprocess


def vggModel(X_train, y_train, X_Test, y_test):
    num_classes = 3

    model = Sequential()
    model.add(Conv2D(32, kernel_size=1, strides=1,
                     activation='relu',
                     padding='same',
                     input_shape=(48, 48, 1)))
    model.add(Conv2D(32, 5, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(32, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()

    model.compile(
        loss=categorical_crossentropy,
        optimizer='sgd',
        metrics=['accuracy']
    )

    model.fit(X_train, y_train, epochs=50, batch_size=128)

    score = model.evaluate(X_Test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

preprocess = Preprocess()
X, y = preprocess.get_all_data()
kf = KFold(n_splits=10, shuffle=True)
for train_index, test_index in kf.split(X):
    train_X, train_y = X[train_index], y[train_index]
    test_X, test_y = X[test_index], y[test_index]

    train_X = train_X.reshape(train_X.shape[0], 48, 48, 1)
    test_X = test_X.reshape(test_X.shape[0], 48, 48, 1)
    train_y = train_y.reshape(train_y.shape[0])
    train_y = to_categorical(train_y, 3)
    test_y = test_y.reshape(test_y.shape[0])
    test_y = to_categorical(test_y, 3)

    vggModel(train_X, train_y, test_X, test_y)
