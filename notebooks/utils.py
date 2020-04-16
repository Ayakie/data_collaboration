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

def make_anchors(Xtrain, nanc):
    
    anc = np.random.uniform(low=np.min(Xtrain), high=np.max(Xtrain), size=(nanc, Xtrain.shape[1]))
    
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


