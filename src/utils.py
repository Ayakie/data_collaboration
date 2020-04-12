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
    
    X_test = X_test[:args.ntest]
    label_test = label_test[:args.ntest]

    return X_train, label_train, X_test, label_test, user_list


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
    print('Training loss: ', dc_score[0])
    print('Test loss: ', dc_score[1])
    print('Test accuracy: ', dc_score[2])

    print('=== Federated Learning ===')
    print('Training loss of FL: ', fl_score[0])
    print('Test loss of FL: ', fl_score[1])
    print('Test accuracy of FL: ', fl_score[2])
    return

def get_3col_plt(centr_score, ind_score, dc_score, fl_score, args):
    '''
    Prameters
    ---------
    score: list
        score[0]: training loss
        score[1]: test loss
        score[2]: test accuracy
    '''
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    ax[0].plot(centr_score[0], marker='.', label='centralized')
    ax[0].plot(ind_score[0], linestyle='-.', marker='.', label='individual')
    ax[0].plot(dc_score[0], marker='.', label='data collaboration')
    ax[0].plot(fl_score[0], marker='.', label='federated learning')
    ax[0].set_xlabel('Rounds for FL, Epochs for the other methods')
    ax[0].legend()
    ax[0].set_title('Training Loss, NN')

    ax[1].plot(centr_score[1], marker='.', label='centralized')
    ax[1].plot(ind_score[1], linestyle='-.', marker='.', label='individual')
    ax[1].plot(dc_score[1], marker='.', label='data collaboration')
    ax[1].plot(fl_score[1], marker='.', label='federated learning')
    ax[1].set_xlabel('Rounds for FL, Epochs for the other methods')
    ax[1].legend()
    ax[1].set_title('Validation Loss, NN')

    ax[2].plot(centr_score[2], marker='.', label='centralized')
    ax[2].plot(ind_score[2], linestyle='-.', marker='.', label='individual')
    ax[2].plot(dc_score[2], marker='.', label='data collaboration')
    ax[2].plot(fl_score[2], marker='.', label='federated learning')
    ax[2].set_xlabel('Rounds for FL, Epochs for the other methods')
    ax[2].legend()
    ax[2].set_title('Validation Accuracy, NN')

    dir_path = "/Users/nedo_m02/Desktop/pytorch_practice/FL"
    if args.save_fig:
        plt.savefig(dir_path + '/save/figures/fed_dc_%s_%sanc_%sir_%susers_iid[%s]_%sround_%srun.png'
                    % (args.dataset, args.nanc, args.d_ir, args.num_users, args.iid, args.nround, args.repeat))
    else:
        pass
    plt.show()