import numpy as np
import random
from collections import Counter
import time
from tqdm import tqdm
from utils import get_dataset, knn_cls, exp_details
from options import args_parser
from dc_utils import data_collaboration

# graph
import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_palette('Set2')
sns.set_style('darkgrid')
sns.set_context(font_scale=1.3)

'''args parameters
num_users = 10            # number of parties
ndat = 100                # size of data portions
nanc = 2000               # number of anchor data
n_neighbors = 6           # for LLE, LPP
d = 75                    # dimension of intermediate representation
repeat = 5                # number of repeat to experiment(≒epochs)
anc_type = 'random'  
'''
args = args_parser()
repeat = args.repeat                  # number of repeated experiments
n_components = args.d       # dimension of target unified space
method = 'PCA'              # method to make IR

if __name__ == '__main__':
    acc_cntr = np.zeros([repeat, args.num_users])
    acc_ind = np.zeros([repeat, args.num_users])
    acc_dc = np.zeros([repeat, args.num_users])

    start_time= time.time()

    for r in range(repeat):
        print(f'Round {r+1}')
        random.seed(args.seed)
        X_train, label_train, X_test, label_test, Xanc, user_list = get_dataset(args)
        
        # do not define here but inside the roop below, row 51
        # user_list_ii = user_list[ii]
        for ii in tqdm(range(args.num_users)):
            print(f'User {ii+1}: {Counter(label_train[user_list[ii]])}')

            # pseudo-split of data
            Div_data = []
            for i in range(ii+1):
                user_idx_i = user_list[i] # caution: row 43
                Div_data.append({'X':X_train[user_idx_i], 'Xtest':X_test, 'Xanc':Xanc,
                                 'label_train': label_train, 'label_test':label_test,}) 

            # Main
            # train classifier on all data
            idx_all = np.ravel(user_list[:ii+1]).tolist()
            X_all = X_train[idx_all]
            label_all = label_train[idx_all]
            cntr = knn_cls(X_all, X_test, label_all, label_test)
            acc_cntr[r, ii] = cntr
            
            if ii == 0:
                # train classifier on data of one user
                ind = cntr
            acc_ind[r, ii] = ind

            # Proposed method
            X_dc, X_test_dc = data_collaboration(Div_data, method, args)
            dc = knn_cls(X_dc, X_test_dc, label_all, label_test)
            acc_dc[r, ii] = dc
            
    end_time = time.time()
    print('Time for computation: ', end_time - start_time)
    knn_cntr = np.around(np.mean(acc_cntr, 0), decimals=2)
    knn_ind = np.around(np.mean(acc_ind, 0), decimals=2)
    knn_dc = np.around(np.mean(acc_dc, 0), decimals=2)

    print('Averaged over {} runs'.format(repeat))
    print('Centralized average accuracy:', knn_cntr)
    print('Individual average accuracy:', knn_ind)
    print('Collaboration average accuracy:', knn_dc)

    dir_path = "/Users/nedo_m02/Desktop/pytorch_practice/FL/"
    plt.figure(figsize=(13,5))
    plt.plot(knn_cntr, label='Centralized')
    plt.plot(knn_ind, label='Indivisual (User1)')
    plt.plot(knn_dc, label='Data Collaboration')
    plt.xlabel('Number of users')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of {} ({})'.format(args.dataset, ['IID' if args.iid==1 else 'Non-IID'][0]))
    plt.legend()
    plt.savefig(dir_path + 'save/figures/dc_%sanc_%susers_iid[%s]_%srepeat.png'%(args.nanc, args.num_users, args.iid, repeat))
    plt.show()
