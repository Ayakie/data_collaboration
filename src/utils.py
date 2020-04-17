import numpy as np
from models import GlobalModel
from sampling import make_anchors, sampling_iid, sampling_noniid
from keras.datasets import mnist, cifar10, fashion_mnist
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

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

def exp_results(centr_score, ind_score, dc_score, fl_score, args):
    print(f'Averaged over {args.repeat} runs')
    print('=== Centralized Model ===')
    print('Training loss: ', centr_score[0])
    print('Test loss: ', centr_score[1])
    print('Test accuracy: ', centr_score[2])

    print('=== Individual Model ===')
    print('Training loss: ', ind_score[0])
    print('Test loss: ', ind_score[1])
    print('Test accuracy: ', ind_score[2])

    print('=== Data Collaboration ===')
    if len(dc_score) == 3:
        print('Training loss: ', dc_score[0])
        print('Test loss: ', dc_score[1])
        print('Test accuracy: ', dc_score[2])

    elif len(dc_score) == 2:
        print(f'Training loss: {args.anc_type} - {dc_score[0][0]}, random - {dc_score[1][0]}')
        print(f'Test loss: {args.anc_type} - {dc_score[0][1]}, random - {dc_score[1][1]}')
        print(f'Test accuracy: {args.anc_type} - {dc_score[0][2]}, random - {dc_score[1][2]}')

    else:
        raise Exception('Unrecognized shape')       

    print('=== Federated Learning ===')
    print('Training loss of FL: ', fl_score[0])
    print('Test loss of FL: ', fl_score[1])
    print('Test accuracy of FL: ', fl_score[2])
    return

def get_3col_plt(centr_score, ind_score, dc_score, fl_score, args, xlabel, x_val):
    '''
    Prameters
    ---------
    score: list
        score[0]: training loss
        score[1]: test loss
        score[2]: test accuracy
    '''
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    x_val = np.array(x_val)

    ax[0].plot(x_val, centr_score[0], marker='.', label='centralized')
    ax[0].plot(x_val, ind_score[0], linestyle='-.', marker='.', label='individual')
    ax[0].plot(x_val, dc_score[0], marker='.', label='data collaboration')
    ax[0].plot(x_val, fl_score[0], marker='.', label='federated learning')
    ax[0].legend()
    ax[0].set_title('Training Loss, NN')

    ax[1].plot(x_val, centr_score[1], marker='.', label='centralized')
    ax[1].plot(x_val, ind_score[1], linestyle='-.', marker='.', label='individual')
    ax[1].plot(x_val, dc_score[1], marker='.', label='data collaboration')
    ax[1].plot(x_val, fl_score[1], marker='.', label='federated learning')
    ax[1].legend()
    ax[1].set_title('Validation Loss, NN')

    ax[2].plot(x_val, centr_score[2], marker='.', label='centralized')
    ax[2].plot(x_val, ind_score[2], linestyle='-.', marker='.', label='individual')
    ax[2].plot(x_val, dc_score[2], marker='.', label='data collaboration')
    ax[2].plot(x_val, fl_score[2], marker='.', label='federated learning')
    ax[2].legend()
    ax[2].set_title('Validation Accuracy, NN')

    for ax in ax.flat:
        ax.set(xlabel=xlabel)

    if args.save_fig:
        plt.savefig('save/figures/fed_dc_%s_%sanc_%s_%sir_%susers_iid[%s]_%sround_%srun.png'
                    % (args.dataset, args.nanc, args.anc_type, args.d_ir, args.num_users, args.iid, args.nround, args.repeat))
    else:
        pass
    plt.show()

def get_3col_plt_2dc(centr_score, ind_score, dc_score, fl_score, args, xlabel, x_val):
    '''
    Prameters
    ---------
    score: list
        score[0]: training loss
        score[1]: test loss
        score[2]: test accuracy
    '''
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    x_val = np.array(x_val)

    ax[0].plot(x_val, centr_score[0], marker='.', label='centralized')
    ax[0].plot(x_val, ind_score[0], linestyle='-.', marker='.', label='individual')
    ax[0].plot(x_val, dc_score[0][0], marker='.', label=f'data collaboration({args.anc_type})', color=colormap2[-1])
    ax[0].plot(x_val, dc_score[1][0], marker='.', label='data collaboration(random)')
    ax[0].plot(x_val, fl_score[0], marker='.', label='federated learning')
    ax[0].legend()
    ax[0].set_title('Training Loss, NN')

    ax[1].plot(x_val, centr_score[1], marker='.', label='centralized')
    ax[1].plot(x_val, ind_score[1], linestyle='-.', marker='.', label='individual')
    ax[1].plot(x_val, dc_score[0][1], marker='.', label=f'data collaboration({args.anc_type})', color=colormap2[-1])
    ax[1].plot(x_val, dc_score[1][1], marker='.', label='data collaboration(random)')
    ax[1].plot(x_val, fl_score[1], marker='.', label='federated learning')
    ax[1].legend()
    ax[1].set_title('Validation Loss, NN')

    ax[2].plot(x_val, centr_score[2], marker='.', label='centralized')
    ax[2].plot(x_val, ind_score[2], linestyle='-.', marker='.', label='individual')
    ax[2].plot(x_val, dc_score[0][2], marker='.', label=f'data collaboration({args.anc_type})', color=colormap2[-1])
    ax[2].plot(x_val, dc_score[1][2], marker='.', label='data collaboration(random)')
    ax[2].plot(x_val, fl_score[2], marker='.', label='federated learning')
    ax[2].legend()
    ax[2].set_title('Validation Accuracy, NN')

    for ax in ax.flat:
        ax.set(xlabel=xlabel)

    if args.save_fig:
        plt.savefig('save/figures/nusers_fed_dc_%s_%sanc_%s_%sir_%susers_iid[%s]_%sround_%srun.png'
                    % (args.dataset, args.nanc, args.anc_type, args.d_ir, args.num_users, args.iid, args.nround, args.repeat))
    else:
        pass
    plt.show()
