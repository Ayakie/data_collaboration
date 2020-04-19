import numpy as np
import random
from collections import Counter
import time
from tqdm import tqdm
from utils import get_dataset, exp_details, fed_avg
from models import GlobalModel
from sampling import make_anchors
from options import args_parser
from dc_utils import data_collaboration
import datetime

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
nanc = 500                # number of anchor data
n_neighbors = 6           # for LLE, LPP, KNN
d_ir = 50                 # dimension of intermediate representation(half of ndat)
repeat = 3                # number of repeat to experiment(epochs)
'''

args = args_parser()
ir_method = 'SVD'         # method to make IR

if __name__ == '__main__':
    acc_cntr = np.zeros([args.repeat, args.num_users])
    acc_ind = np.zeros([args.repeat, args.num_users])
    acc_dc = np.zeros([args.repeat, args.num_users])

    time_dc = np.zeros([args.repeat, args.num_users])

    for r in range(args.repeat):
        print(f'Repeat: {r+1} / {args.repeat}')
        random.seed(r)
        X_train, X_test, X_anc, label_train, label_test, user_list = get_dataset(args)

        for ii in tqdm(range(args.num_users)):
            print(f"User {ii+1}: {Counter(label_train[user_list[ii]])}")

            # Main
            # train classifier on all data
            idx_all = np.ravel(user_list[:ii+1]).tolist()
            X_all = X_train[idx_all]
            label_all = label_train[idx_all]
            num_class = len(np.unique(label_test))

            centr_model = GlobalModel(args, X_all, num_class).set_model()

            if args.model == 'knn' or args.model == 'svm':
                centr_model.fit(X_all, label_all)
                centr = centr_model.score(X_test, label_test)
            else:  # keras model
                centr_model.fit(
                    X_all, label_all, batch_size=args.batch_size, epochs=args.nround, verbose=0)
                centr = centr_model.evaluate(X_test, label_test)[1]
            acc_cntr[r, ii] = centr

            if ii == 0:
                # train classifier on data of one user
                ind = centr
            acc_ind[r, ii] = ind

            # Proposed method(User 1 has test data)
            # pseudo-split of data

            start = time.time()

            anc = make_anchors(X_anc)

            Div_data = []
            for i in range(ii+1):
                user_idx_i = user_list[i]
                Div_data.append(
                    {'X': X_train[user_idx_i], 'Xtest': X_test, 'anc': anc})
            
            X_dc, X_test_dc, _ = data_collaboration(Div_data, ir_method, args)

            assert len(X_dc) == len(X_all)

            dc_model = GlobalModel(args, X_dc, num_class).set_model()

            if args.model == 'knn' or args.model == 'svm':
                dc_model.fit(X_dc, label_all)
                dc = dc_model.score(X_test_dc, label_test)
            else:
                dc_model.fit(X_dc, label_all, batch_size=args.batch_size,
                             epochs=args.nround, verbose=0)
                dc = dc_model.evaluate(X_test_dc, label_test)[1]
            
            acc_dc[r, ii] = dc

            end = time.time()
            time_dc[r, ii] = end - start

    centr = np.round(np.mean(acc_cntr, 0), decimals=3)
    ind = np.round(np.mean(acc_ind, 0), decimals=3)
    dc = np.round(np.mean(acc_dc, 0), decimals=3)

    print('Averaged over {} runs'.format(args.repeat))
    print('Centralized average accuracy:', centr)
    print('Individual average accuracy:', ind)
    print('Collaboration average accuracy:', dc)

    xval = np.arange(1, args.num_users+1)
    plt.figure(figsize=(13, 5))
    plt.plot(xval, centr, label='Centralized', marker=".")
    plt.plot(xval, ind, label='Indivisual (User1)', marker=".")
    plt.plot(xval, dc, label='Data Collaboration', marker=".")
    plt.xlabel('Number of users')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of {} ({})'.format(
        args.dataset.upper(), ['IID' if args.iid else 'Non-IID'][0]))
    plt.ylim(min(dc)-0.1, max(centr)+0.1)
    plt.legend()
    if args.save_fig:
        plt.savefig('./save/figures/dc_%s_%s_%s_%srun_%s_users.png' % (
            args.dataset, args.model, args.anc_type, args.repeat, ir_method))
        
        # save data as numpy
        np.savez('./save/logs/dc_%s_%s_%s_%srun_%s_users' % (
            args.dataset, args.model, args.anc_type, args.repeat, ir_method), 
            cntr=acc_cntr, ind=acc_ind, dc=acc_dc, time_dc=time_dc)
    else:
        pass
    plt.show()

    try:
        with open('./save/logs/dc_%s_%s_%s_%srun_%s_users.txt' % (
            args.dataset, args.model, args.anc_type, args.repeat, ir_method), 'w') as log:
            print(args, file=log)
            print(centr, file=log)
            print(ind, file=log)
            print(dc, file=log)

    except IOError:
        print('File Error')
