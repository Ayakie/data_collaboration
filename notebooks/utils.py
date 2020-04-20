import numpy as np
import pandas as pd
import numpy.matlib
import scipy as sc
import random
import datetime
import tensorflow as tf
from keras.datasets import mnist, fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPool2D, BatchNormalization
from keras.metrics import sparse_categorical_accuracy
from keras.optimizers import SGD, RMSprop, Adadelta, Adam

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import LeakyReLU, BatchNormalization
from keras.optimizers import Adam
from keras.models import Sequential, Model

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.preprocessing import normalize, LabelBinarizer
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm

import seaborn as sns
sns.set_palette('Set2', n_colors=5)
sns.set_style('darkgrid')
sns.set_context(font_scale=1.3)

import sys, os
sys.path.append('..')
from packages.lpproj_LPP import LocalityPreservingProjection as LPP

def sampling_iid(X_train, num_users, ndat):
    '''Sample I.I.D user data from dataset
    Return
    --------
    list of user index(key) and ndat samples index(value) 
    '''

    list_users, all_idx = [], [i for i in range(len(X_train))]
    for i in range(num_users):
        user_idx = set(np.random.choice(all_idx, size=ndat, replace=False))
        all_idx = list(set(all_idx) - user_idx) # remained index
        list_users.append(list(user_idx))
    
    return list_users

# Simple KNN Classifier
def knn_cls(Xtrain, Xtest, ytrain, ytest): 
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(Xtrain, ytrain.ravel())
    score = model.score(Xtest, ytest.ravel())
    
    return score


# essential functions

def get_dc(colab, n_components):
    
    anc_merged = np.hstack([i['Yanc'] for i in colab])

    U, S, V = sc.linalg.svd(anc_merged, lapack_driver='gesvd')
    Z = U.T[:n_components,:]
    
    unified = []
    for i, user in enumerate(colab):
        G = np.dot(Z, np.linalg.pinv(user['Yanc'].T))
        mapping = np.dot(user['Y'], G.T)                   
        unified.append(mapping)
        # first user has training data
        if i == 0:
            transformed_test = np.dot(user['Ytest'], G.T)
    
    return unified, transformed_test

def get_embedding(X, anc, Xtest, d, method='SVD'):
    
    if method == 'PCA':
        F = PCA(d, svd_solver='full')
    
    elif method == 'ICA':
        F = FastICA(d)
        
    # elif method == 'SRP':
    #     F = SparseRandomProjection(d)
    
    elif method == 'LLE':
        F = LocallyLinearEmbedding(n_neighbors=15, n_components=d)
 
    else:
        U1,S1,V1 = sc.linalg.svd(X, lapack_driver='gesvd')
        F = V1[:d,:]
        Y = np.dot(X, F.T)
        Yanc = np.dot(anc, F.T)
        Ytest = np.dot(Xtest, F.T)
        
        return Y, Yanc, Ytest
        
    Y = F.fit_transform(X)
    Yanc = F.transform(anc)
    Ytest = F.transform(Xtest)
    
    return Y, Yanc, Ytest

def make_anchors(X_train, nanc, anc_type, hidden_dim=128, epochs=300, latent_dim=100):
    '''
    Parameters
    --------------
    nanc: int
        number of anchor data
    X_train: 
        training data to generate anchor data
    anc_type: 'random', 'gan', 'saved', 'raw'
        'gan_new': generate anchor data by GAN from X_train to generate anchor data
        'gan': load saved weight of anchor generator previously trained by anc_type='gan_new'
        'raw': sample from original data(X_train) as anchors
    
    '''
    if anc_type == 'random':
    
        anc = np.random.uniform(low=np.min(X_train), high=np.max(X_train), size=(nanc, X_train.shape[1]))
        # anc = normalize(anc)
        # anc = np.random.uniform(0, 255, size=(nanc, X_train.shape[1]))
    
    elif anc_type == 'gan_new':

        gan = GAN(input_shape=X_train.shape[1], hidden_dim=hidden_dim, latent_dim=latent_dim)
        gan.train(X_train, epochs=epochs, batch_size=32)
        
        noise = np.random.normal(0, 1, size=(nanc, latent_dim))
        anc = gan.generator.predict(noise)

        # save model
        gan.generator.save_weights('save/models/generator_%sdata_%sepochs_%sanc.h5' % (len(X_train), epochs, nanc) )
    
    elif anc_type == 'gan':

        noise = np.random.normal(0, 1, size=(nanc, latent_dim))
        generator = GAN(X_train.shape[1], hidden_dim, latent_dim).generator
        generator.load_weights('save/models/generator_1200data_2000epochs_500anc.h5')
        # generator.load_weights('save/models/generator_%sdata_%sepochs_%sanc.h5' % (len(X_train), epochs, nanc) )

        anc = generator.predict(noise)
    
    elif anc_type == 'raw':
        
        idx = np.random.choice(len(X_train), nanc, replace=False)
        anc = X_train[idx]
    
    else:
        raise Exception('Unrecognized type')
    
    return anc


def merge_anchors(Yanc1, Yanc2, dd):
    
    anc_merged = np.hstack([Yanc1, Yanc2])

    U, S, V = sc.linalg.svd(anc_merged, lapack_driver='gesvd')
    Z = U.T[:dd,:]

    G1 = np.dot(Z, np.linalg.pinv(Yanc1.T))
    anc1_h = np.dot(G1, Yanc1.T)
    G2 = np.dot(Z, np.linalg.pinv(Yanc2.T))
    anc2_h = np.dot(G2, Yanc2.T)
    
    return Z.T, anc1_h.T, anc2_h.T

def data_collaboration(Div_data, dd):
    colab = []
    for user in Div_data:
        Y, Yanc, Ytest = get_embedding(user['X'], user['anc'], user['Xtest'], dd, method='PCA')
        colab.append({'Y':Y, 'Yanc':Yanc, 'Ytest':Ytest })
    
#     # check anchor allignment
#     _, anch1, anch2 = merge_anchors(colab[0]['Yanc'], colab[1]['Yanc'], dd)
#     plt.scatter(anch1[:20,0], anch1[:20,1])
#     plt.scatter(anch2[:20,0], anch2[:20,1])
#     plt.title('Anchor allignment')

        
    unified_data, transformed_test = get_dc(colab, dd)
    X_all = np.vstack(unified_data)
    
    return X_all, transformed_test


def create_compiled_keras_model(dd):
    model = tf.keras.models.Sequential([
        
        tf.keras.layers.Dense(512,activation="relu",input_shape=(dd,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
    
    def loss_fn(y_true, y_pred):
        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
        y_true, y_pred))
    
    model.compile(
      loss=loss_fn,
      optimizer="adam",
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    
    return model


class GAN():
    def __init__(self, input_shape, hidden_dim=256, latent_dim=100, optimizer=Adam(0.0002, 0.5)):
        self.input_shape = (input_shape,)
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # define generator instance
        self.generator = self.build_generator()
        
        
        # defince discriminator instance
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
        
        # defined combined network
        self.combined = self.build_combined()
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
        
    
    # generator model
    def build_generator(self):
        
        model = Sequential()
        
        model.add(Dense(self.hidden_dim, input_shape=(self.latent_dim,))) # 100 -> 256
        model.add(LeakyReLU(0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.hidden_dim*2)) # 256 -> 512
        model.add(LeakyReLU(0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.hidden_dim*4)) # 512 -> 1024
        model.add(LeakyReLU(0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.input_shape), activation='tanh'))
#         model.add(Reshape(self.input_shape))
        
        model.summary()
        
        return model
    
    # discriminator model
    def build_discriminator(self):
        
        model = Sequential()
        
#         model.add(Flatten(input_shape=self.input_shape))
#         AttributeError: 'tuple' object has no attribute 'lower'
        model.add(Dense(self.hidden_dim*2, input_shape=self.input_shape)) # 256 -> 512
        model.add(LeakyReLU(0.2))
        model.add(Dense(self.hidden_dim)) # 512 -> 256
        model.add(LeakyReLU(0.2))
        model.add(Dense(1, activation='sigmoid')) # 256 -> number of class (0 or 1)
        
        model.summary()
        
        return model
    
    # generatorとdiscriminatorを結合させる
    
    def build_combined(self):
        self.discriminator.trainable = False
        model = Sequential([self.generator, self.discriminator])
        return model
        
    
    def train(self, X_train, epochs, batch_size=32):
        
        half_batch = int(batch_size / 2)
        
        num_batches = int(X_train.shape[0] / half_batch)
        print('Number of batches:', num_batches)
        
        
        for epoch in range(epochs+1):
            
            for iteration in range(num_batches):
            
                # ------ discriminator --------

                # pick up half-batch size of datafrom generator: G(x)
                noise = np.random.normal(0, 1, (half_batch, self.latent_dim))
                gen_data = self.generator.predict(noise)

                # pick up half-batch size of dataset from dataset: D(x)
                idx = np.random.choice(X_train.shape[0], half_batch)
                true_data = X_train[idx]

                # make label for training: gen_data=0, true_data=1
                y_gen = np.zeros((half_batch, 1))
                y_true = np.ones((half_batch, 1))

                # learn discriminator
                d_loss_fake = self.discriminator.train_on_batch(gen_data, y_gen)
                d_loss_real = self.discriminator.train_on_batch(true_data, y_true)

                # average each loss
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                
                # ------- generator ---------

                noise = np.random.normal(0, 1, (half_batch, self.latent_dim))
                
                # make label
                valid_y = np.ones((half_batch, 1))
                # valid_y = np.array([1] * half_batch)

                # train generator
                g_loss = self.combined.train_on_batch(noise, valid_y)
                
            # progress
            print('Epoch: %d, [D loss: %f] [G loss: %f]' % (epoch, d_loss, g_loss))

            if epoch % 200 == 0:
                self.save_fig(X_train, epoch)

    # generate 25 figures during training
    def save_fig(self, X_train, epoch):
        noise = np.random.normal(0, 1, size=(25, self.latent_dim))
        gen_img = self.generator.predict(noise)

        fig, axs = plt.subplots(5,5, figsize=(5,5))

        for i in range(5):
            for j in range(5):
                axs[i, j].imshow(gen_img[i*5+j].reshape(28, 28), cmap='gray')
        
        fig.savefig('save/figures/%sdata_%sepoch.png' % (len(X_train), epoch))
        plt.close()

def fed_avg(user_weight_list, user_ndat_list):
    '''conpute FedAvg: weighted avarage of parameters.
    if the size of data is equal among users, fedavg will be the normal avarage

    Return
    ----------
    new_weights: list
        list of new weights per each user
    
    '''
    new_weight = [np.zeros(w.shape) for w in user_weight_list[0]] # get shape of params
    total_size = np.sum(user_ndat_list)

    for w in range(len(new_weight)):
        for i in range(len(user_weight_list)): # num_user
            new_weight[w] += user_weight_list[i][w] * user_ndat_list[i] / total_size

    return new_weight