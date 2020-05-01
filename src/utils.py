import numpy as np
import pandas as pd
from models import GlobalModel
from sampling import make_anchors, sampling_iid, sampling_noniid
from keras.datasets import mnist, cifar10, fashion_mnist
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import sys, os
import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import csv

import matplotlib.pyplot as plt
from options import args_parser

import seaborn as sns
colormap2 = sns.color_palette('RdBu')

args = args_parser()

def get_dataset(args, nlabel=2, anc_size=0.02, ndat=args.ndat, num_users=args.num_users):
    '''Return train, anchor, test datasets and user group list

    Parameters
    ------------
    args: args_parser()
    nlabel: int(2~9)
        How many labels each user has 
        reference: https://github.com/AshwinRJ/Federated-Learning-PyTorch
    
    anc_size: default(0.02 x 60000 = 1200 data)
        ratio to train dataset to generate anchor data

    '''

    if args.dataset == 'mnist':
        (X_train, label_train), (X_test, label_test) = mnist.load_data()
    
    elif args.dataset == 'fashion_mnist':
        (X_train, label_train), (X_test, label_test) = fashion_mnist.load_data()

    elif args.dataset == 'cifar':
        (X_train, label_train), (X_test, label_test) = cifar10.load_data()
        label_train = label_train.ravel()
        label_test = label_test.ravel()
        
    else:
        raise Exception('Passed args')
    
    # pick up generating anchor data from train data
    X_train, X_anc, label_train, _ = train_test_split(X_train, label_train, test_size=anc_size, random_state=42, shuffle=True)

    if args.iid:
        user_list = sampling_iid(X_train, num_users, ndat)
    else:
        user_list = sampling_noniid(X_train, label_train, num_users, ndat, nlabel=nlabel)        
    

    # reshape
    X_train = X_train.reshape(X_train.shape[0], -1) / 255
    X_test = X_test.reshape(X_test.shape[0], -1) / 255
    X_anc = X_anc.reshape(X_anc.shape[0], -1) / 255
    
    # pick up test data
    X_test = X_test[:args.ntest]
    label_test = label_test[:args.ntest]

    return X_train, X_test, X_anc, label_train, label_test, user_list


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


def get_result(xval, acc_cntr, acc_ind, acc_method, time, args, setting='users', method='dc'):
    
    '''
    Parameters
    ---------
    setting: 'users' or 'ndat'
        record the parameter setting. select 'users' for dc_main.py and fed_main.py
        selct 'ndat' for dc_ndat.py and fed_ndat.py

    method: 'dc' or 'fed'
        select 'dc' for the data collabortaion experiment else 'fed'.

    Returns
    --------
    file name: (day)_(hour)_(minutes)_(method)_(data)_(model)_(acn_type)_(repeat)_(setting)
    ex) 0420_18_11_dc_mnist_mlp_random_10run_users.txt
    '''

    centr_m = np.round(np.mean(acc_cntr, 0), decimals=3)
    centr_s = np.round(np.std(acc_cntr, 0), decimals=3)
    ind_m = np.round(np.mean(acc_ind, 0), decimals=3)
    ind_s = np.round(np.std(acc_ind, 0), decimals=3)
    other_m = np.round(np.mean(acc_method, 0), decimals=3)
    other_s = np.round(np.std(acc_method, 0), decimals=3)
    time_m = np.round(np.mean(time, 0), decimals=3)
    time_s = np.round(np.std(time, 0), decimals=3)

    print('Averaged over {} runs'.format(args.repeat))
    print('Centralized average accuracy:', centr_m)
    print('Individual average accuracy:', ind_m)
    print(['Collaboration average accuracy:' if method=='dc' else 'Federated Learning'][0], other_m)

    plt.figure(figsize=(13, 5))
    plt.plot(xval, centr_m, label='Centralized', marker=".")
    plt.plot(xval, ind_m, label='Indivisual (User1)', marker=".")
    plt.plot(xval, other_m, label=['Data Collaboration' if method=='dc' else 'Federated Learning'][0], marker=".")
    plt.xlabel(['Number of users' if setting=='users' else 'Number of data per user'][0])
    plt.ylabel('Accuracy')
    plt.title('Accuracy of {} ({})'.format(
        args.dataset.upper(), ['IID' if args.iid else 'Non-IID'][0]))
    # plt.ylim(min(other)-0.1, max(centr)+0.05)
    plt.legend()

    now = datetime.datetime.now()
    save_time = now.strftime("%m%d_%H_%M")

    if args.save_fig:
        plt.savefig('./save/figures/%s_%s_%s_%susers_%s_%s_%sE_%srun_%s.png' % (
            save_time, method, args.dataset, args.num_users, args.model, args.anc_type, args.epoch, args.repeat, setting))
        
    else:
        pass
    plt.show()

    if args.save_log:
        # save as csv
        df = pd.DataFrame(columns=xval)
        df = df.append(pd.Series(args, name='args'))
        df = df.append(pd.Series(centr_m, name='centr_m', index=xval))
        df = df.append(pd.Series(centr_s, name='centr_s', index=xval))
        df = df.append(pd.Series(ind_m, name='ind_m', index=xval))
        df = df.append(pd.Series(ind_s, name='ind_s', index=xval))
        df = df.append(pd.Series(other_m, name=method+'_m', index=xval))
        df = df.append(pd.Series(other_s, name=method+'_s', index=xval))
        df.to_csv('./save/logs/%s_%s_%s_%susers_%s_%s_%sE_%srun_%s.csv' % (
            save_time, method, args.dataset, args.num_users, args.model, args.anc_type,  args.epoch, args.repeat, setting))
