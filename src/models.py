import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import SGD, Adadelta, Adam, Adamax
from keras.metrics import sparse_categorical_accuracy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier


class GlobalModel(object):
    optimizer_dict = {'sgd': SGD(), 'adamax': Adamax(),
                  'adadelta': Adadelta(), 'adam': Adam()}

    def __init__(self, args, X_train, num_class):
        self.args = args
        self.input_shape = X_train.shape[1:]
        self.num_class = num_class
        self.optimizer = self.optimizer_dict[self.args.optimizer]
    
    def set_model(self):
        if self.args.model == 'cnn':
            if self.args.dataset == 'mnist':
                model = self.cnn_mnist()
            elif self.args.dataset == 'fashion_mnist':
                model = self.cnn_fashion_mnist()
            elif self.args.dataset == 'cifar':
                model = self.cnn_cifar()
        elif self.args.model == 'mlp':
            model = self.mlp()
        elif self.args.model == 'knn':
            model = self.knn_cls()
        else:
            raise Exception('Passed args')
        
        return model

    def mlp(self):

        model = Sequential()
        # kernel_initializer='random_uniform'
        model.add(Dense(512, activation='relu',
                        input_shape=self.input_shape))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.num_class, activation='softmax'))

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=self.optimizer, metrics=[sparse_categorical_accuracy])

        return model

    def cnn_mnist(self):

        model = Sequential()
        model.add(Conv2D(10, kernel_size=(5, 5), activation='relu',
                         input_shape=self.input_shape))  # 1x28x28 -> 10x24x24
        model.add(MaxPooling2D(pool_size=(2, 2)))  # 10x24x24 -> 10x12x12
        # 10x12x12 -> 20x8x8
        model.add(Conv2D(20, kernel_size=(5, 5), activation='relu'))
        model.add(Dropout(0.25))
        model.add(MaxPooling2D(pool_size=(2, 2)))  # 20x8x8 -> 20x4x4
        model.add(Flatten())  # 20x4x4 -> 20*4*4
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(self.num_class, activation='softmax'))

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=self.optimizer, metrics=[sparse_categorical_accuracy])

        return model

    def cnn_fashion_mnist(self):

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                         input_shape=self.input_shape))  # 32x26x26
        model.add(Conv2D(64, (3, 3), activation='relu'))  # 64x24x24
        model.add(MaxPooling2D((2, 2)))  # 64x12x12
        model.add(BatchNormalization())
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_class, activation='softmax'))

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=self.optimizer, metrics=[sparse_categorical_accuracy])

        return model

    def cnn_cifar(self):

        model = Sequential()
        model.add(Reshape((32, 32, 3), input_shape=self.input_shape))
        model.add(Conv2D(32, kernel_size=(5, 5), padding='same',
                         activation='relu', input_shape=self.input_shape))  # 32x32x32
        model.add(MaxPooling2D((2, 2)))  # 32x16x16
        model.add(Dropout(0.25))
        model.add(Conv2D(64, (5, 5), padding='same',
                         activation='relu'))  # 64x16x16
        model.add(MaxPooling2D(2, 2))  # 64x8x8
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_class, activation='softmax'))

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=self.optimizer, metrics=[sparse_categorical_accuracy])

        return model

    # Simple KNN Classifier
    def knn_cls(self):
        model = KNeighborsClassifier(n_neighbors=self.args.n_neighbors)

        return model
    