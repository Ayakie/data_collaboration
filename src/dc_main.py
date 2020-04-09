import numpy as np
import random
from collections import Counter
import time
from tqdm import tqdm
from utils import get_dataset, exp_details
from models import knn_cls
from options import args_parser
from dc_utils import data_collaboration

# graph
import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_palette('Set2', n_colors=5)
sns.set_style('darkgrid')
sns.set_context(font_scale=1.3)

'''args parameters
num_users = 10            # number of parties
ndat = 100                # size of data portions
nanc = 1000 or 2000       # number of anchor data
n_neighbors = 6           # for LLE, LPP
d_ir = 75                 # dimension of intermediate representation
repeat = 5                # number of repeat to experiment(≒epochs)
anc_type = 'random'  
'''
args = args_parser()
ir_method = 'PCA'           # method to make IR

if __name__ == '__main__':
    acc_cntr = np.zeros([args.repeat, args.num_users])
    acc_ind = np.zeros([args.repeat, args.num_users])
    acc_dc = np.zeros([args.repeat, args.num_users])

    start_time= time.time()

    for r in range(args.repeat):
        print(f'Round {r+1}')
        random.seed(args.seed)
        X_train, label_train, X_test, label_test, Xanc, user_list = get_dataset(args)
        # do not define here but inside the roop below, row 51
        # user_idx_ii = user_list[ii]
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
            X_dc, X_test_dc = data_collaboration(Div_data, ir_method, args)
            dc = knn_cls(X_dc, X_test_dc, label_all, label_test)
            acc_dc[r, ii] = dc
            
    end_time = time.time()
    print('Time for computation: ', end_time - start_time)
    knn_cntr = np.round(np.mean(acc_cntr, 0), decimals=3)
    knn_ind = np.round(np.mean(acc_ind, 0), decimals=3)
    knn_dc = np.round(np.mean(acc_dc, 0), decimals=3)

    print('Averaged over {} runs'.format(args.repeat))
    print('Centralized average accuracy:', knn_cntr)
    print('Individual average accuracy:', knn_ind)
    print('Collaboration average accuracy:', knn_dc)

    dir_path = "/Users/nedo_m02/Desktop/pytorch_practice/FL"
    plt.figure(figsize=(13,5))
    plt.plot(knn_cntr, label='Centralized', marker=".")
    plt.plot(knn_ind, label='Indivisual (User1)', marker=".")
    plt.plot(knn_dc, label='Data Collaboration', marker=".")
    plt.xlabel('Number of users')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of {} ({})'.format(args.dataset.upper(), ['IID' if args.iid else 'Non-IID'][0]))
    plt.ylim([(0.5, 0.85) if args.iid else (0.2, 0.85)][0]) # change the range of yaxis depending on if iid or not
    plt.legend()
    if args.save_fig:
        plt.savefig(dir_path + '/save/figures/dc_%s_%sanc_%susers_iid[%s]_%srun.png'%(args.dataset, args.nanc, args.num_users, args.iid, args.repeat))
    else:
        pass
    plt.show()
