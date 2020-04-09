import copy
import numpy as np
from models import GlobalModel
from sampling import make_anchors, sampling_iid, sampling_noniid
from keras.datasets import mnist, cifar10, fashion_mnist
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

def get_dataset(args, nlabel=2):
    '''Return train and test datasets amd user group which is a dict where
    the keys are the user index and the value are the corresponding data 
    for each of those users.
    '''

    if args.dataset == 'mnist':
        (X_train, label_train), (X_test, label_test) = mnist.load_data()
        # reshape
        X_train = X_train.reshape(X_train.shape[0], -1) / 255
        X_test = X_test.reshape(X_test.shape[0], -1) /255
    
    elif args.dataset == 'fashion_mnist':
        (X_train, label_train), (X_test, label_test) = fashion_mnist.load_data()
        # reshape
        X_train = X_train.reshape(X_train.shape[0], -1) / 255
        X_test = X_test.reshape(X_test.shape[0], -1) / 255

    elif args.dataset == 'cifar':
        (X_train, label_train), (X_test, label_test) = cifar10.load_data()
        label_train = label_train.ravel()
        label_test = label_test.ravel()
        
        # reshape
        X_train = X_train.reshape(X_train.shape[0], -1) / 255
        X_test = X_test.reshape(X_test.shape[0], -1) / 255
    else:
        raise Exception('Passed args')

    if args.iid:
        user_list = sampling_iid(X_train, args.num_users, args.ndat)
    else:
        user_list = sampling_noniid(X_train, label_train, args.num_users, args.ndat, nlabel=nlabel)        

    
    if args.anc_type == 'random':
        Xanc = make_anchors(X_train, args.nanc, args.anc_type)
        Xanc = normalize(Xanc)

    return X_train, label_train, X_test, label_test, Xanc, user_list


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

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Data         : {args.dataset}')
    print(f'    Repeat times : {args.repeat}\n')
    print('     %s' % (['IID' if args.iid else 'Non-IID'][0]))
    print(f'    Number of users   : {args.num_users}')
    print(f'    Anchor    : {args.anc_type}, {args.nanc} size')
    print(f'    Dimension of intermediate representation : {args.d_ir}\n')
    return

def get_3col_plt(centr_hit, ind_hist, dc_hist, fl_hist):
    '''
    Prameters
    ---------
    centr
    '''
    fig, ax = plt.subplots(1, 3, figsize=(18,5))
    ax[0].plot(centr_hit, '.', label='centralized')
    ax[0].plot(ind_hist, '-.', label='individual')
    ax[0].plot(dc_hist, '-+', label='data collaboration')
    ax[1].plot()