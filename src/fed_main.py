import numpy as np
import random
from collections import Counter
import time
from tqdm import tqdm
from utils import get_dataset, exp_details
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
anc_type = 'random'  

args = args_parser()

if __name__ == '__main__':
    acc_cntr = np.zeros([args.repeat, args.num_users])
    acc_ind = np.zeros([args.repeat, args.num_users])
    acc_fed = np.zeros([args.repeat, args.num_users])

    start_time= time.time()

    for r in range(args.repeat):
        print(f"Round {r+1}")
        random.seed(args.seed)
        X_train, label_train, X_test, label_test, user_list = get_dataset(args)
        assert len(label_test) == args.ntest

        for ii in tqdm(range(args.num_users)):
            print(f"User {ii+1}: {Counter(label_train[user_list[ii]])}") 

            # Main
            # train classifier on all data
            idx_all = np.ravel(user_list[:ii+1]).tolist()
            X_all = X_train[idx_all]
            label_all = label_train[idx_all]
            num_class = len(np.unique(label_test))

            centr_model = GlobalModel(args, X_all, num_class).set_model()
            
            
            centr_model.fit(X_all, label_all, batch_size=args.batch_size, epochs=args.nround, verbose=0)
            centr = centr_model.evaluate(X_test, label_test)[1]
            acc_cntr[r, ii] = centr
            
            if ii == 0:
                # train classifier on data of one user
                ind = centr
            acc_ind[r, ii] = ind
            
            # federated learning
            print('----- Federated Learning -----')

            for rr in range(args.nround):
                print(f'==== Round {rr+1} / {args.nround} =====')

                user_w_list, user_ndat_list = [], []
                for c in range(args.num_users):

                    # get a global model and set fedavg weight to it
                    local_model = GlobalModel(args, X_all, num_class).set_model()
                    local_model.set_weights(fl_model.get_weights())  # current_weight

                    local_model.fit(X_train[user_list[c]], label_train[user_list[c]], batch_size=args.batch_size,
                                    epochs=args.epoch, validation_data=(X_test, label_test), verbose=0)

                    # get each trained weight
                    user_w_list.append(local_model.get_weights())
                    # len(Div_data[user]['X'] == ndat
                    user_ndat_list.append(len(X_train[user_list[c]]))

                # calcurate FedAvg
                new_w = fed_avg(user_w_list, user_ndat_list)
                # set new weight to a global model
                fl_model.set_weights(new_w)

                # fl_train_score = fl_model.evaluate(X_all, label_all)
                # fl_test_score = fl_model.evaluate(X_test, label_test)
                
                # fl_score[r, 0, rr] = fl_train_score[0]
                # fl_score[r, 1, rr] = fl_test_score[0]
                # fl_score[r, 2, rr] = fl_test_score[1]
                # # fl_score.append([fl_train_score[0], fl_test_score[0], fl_test_score[1]])
        
                # print(f'Accuracy of FL in round {rr+1}: {fl_test_score[1]}')

            fed = fl_model.evaluate(X_test, label_test)
            acc_fed[r, ii] = fed
            
    end_time = time.time()
    print('Time for computation: ', end_time - start_time)
    centr = np.round(np.mean(acc_cntr, 0), decimals=3)
    ind = np.round(np.mean(acc_ind, 0), decimals=3)
    fed = np.round(np.mean(acc_fed, 0), decimals=3)

    print('Averaged over {} runs'.format(args.repeat))
    print('Centralized average accuracy:', centr)
    print('Individual average accuracy:', ind)
    print('Federated average accuracy:', fed)


    dir_path = "./"
    plt.figure(figsize=(13,5))
    plt.plot(centr, label='Centralized', marker=".")
    plt.plot(ind, label='Indivisual (User1)', marker=".")
    plt.plot(dc, label='Federated Learning', marker=".")
    plt.xlabel('Number of users')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of {} ({})'.format(args.dataset.upper(), ['IID' if args.iid else 'Non-IID'][0]))
    plt.ylim(min(ind)-0.1, max(centr)+0.1)
    plt.legend()
    if args.save_fig:
        plt.savefig(dir_path + '/save/figures/fed_%s_%s_%sanc_%s_dim_%susers_iid[%s]_%srun.png'%(args.dataset, args.model, args.nanc, args.d_ir, args.num_users, args.iid, args.repeat))
    else:
        pass
    plt.show()


    try:
        with open(dir_path + '/save/logs/fed_%s_%s_%sndat_%s_dim.txt'%(args.dataset, args.model, args.ndat, args.d_ir), 'w') as log:
            print(args, file=log)
            print(centr, file=log)
            print(ind, file=log)
            print(fed, file=log)
                
    except IOError:
        print('File Error')
