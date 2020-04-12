import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import SGD, RMSprop, Adadelta, Adam
from keras.models import model_from_config
from keras.metrics import sparse_categorical_accuracy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

# input_shape= X_train.shape[1:]
# num_class=10

class GlobalModel(object):
    optimizer_dict = {'sgd': SGD(), 'rmsprop': RMSprop(),
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
        # model.fit(X_train, label_train.ravel())
        # score = model.score(X_test, label_test.ravel())

        return model
    
    # def gf_cls(self, X_train, X_test, label_train, label_test, lr = self.args.lr, max_depth=self.args.max_depth):
    #     # sklearn default: lr=0.1, max_depth=3
    #     # args default: lr = 0.05, max_depth=10
    #     model = GradientBoostingClassifier(learning_rate=lr, max_depth=max_depth) 

# class LocalUpdate(GlobalModel):
#     def __init__(self, args, input_shape, num_class, local_model, user_idx, div_data):
#         super().__init__(args, input_shape, num_class)
#         self.local_model = local_model
#         self.idx = user_idx
#         self.dataset = div_data

#     def fit(self):
#         self.local_model.fit(self.dataset[self.idx]['X'], self.dataset[self.idx]['label_train'], batch_size=self.args.batch_size, epochs=self.args.epoch, validation_data=)