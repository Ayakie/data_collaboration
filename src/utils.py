import copy
import numpy as np
from sampling import make_anchors, mnist_iid, mnist_noniid
from keras.datasets import mnist
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier

def get_dataset(args, nlabel=2):
    '''Return train and test datasets amd user group which is a dict where
    the keys are the user index and the value are the corresponding data 
    for each of those users.
    '''

    if args.dataset == 'mnist':
        (X_train, label_train), (X_test, label_test) = mnist.load_data()
        
        # reshape
        X_train = normalize(X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
        X_test = normalize(X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2]))

        if args.iid:
            user_list = mnist_iid(X_train, args.num_users, args.ndat)
        else:
            user_list = mnist_noniid(X_train, label_train, args.num_users, args.ndat, nlabel=nlabel)
    
    if args.anc_type == 'random':
        Xanc = make_anchors(X_train, args.nanc, args.anc_type)
        Xanc = normalize(Xanc)

    return X_train, label_train, X_test, label_test, Xanc, user_list

# Simple KNN Classifier
def knn_cls(Xtrain, Xtest, ytrain, ytest): 
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(Xtrain, ytrain.ravel())
    score = model.score(Xtest, ytest.ravel())

    return score

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Data         : {args.dataset}')
    print(f'    Repeat times : {args.repeat}\n')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Number of users   : {args.num_users}')
    print(f'    Type of anchor    : {args.anc_type}')
    print(f'    Number of anchor  : {args.nanc}')
    print(f'    Dimension of intermediate representation : {args.d}\n')
    return