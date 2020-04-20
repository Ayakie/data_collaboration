import numpy as np
import random
from collections import Counter
import time
from tqdm import tqdm
from utils import get_dataset, fed_avg, get_result
from models import GlobalModel
from sampling import make_anchors
from options import args_parser
import datetime

# graph
import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_palette('Set2')
sns.set_style('darkgrid')
sns.set_context(font_scale=1.3)

args = args_parser()

if __name__ == '__main__':
    acc_cntr = np.zeros([args.repeat, args.num_users])
    acc_ind = np.zeros([args.repeat, args.num_users])
    acc_fed = np.zeros([args.repeat, args.num_users])
    time_fed = np.zeros([args.repeat, args.num_users])


    for r in range(args.repeat):
        print(f"Repeat: {r+1} / {args.repeat}")

        random.seed(r)
        X_train, X_test, X_anc, label_train, label_test, user_list = get_dataset(args)
        assert len(label_test) == args.ntest        

        for i in tqdm(range(args.num_users)):
            print(f"User {i+1}: {Counter(label_train[user_list[i]])}")

            # Main
            # train classifier on all data
            idx_all = np.ravel(user_list[:i+1]).tolist()
            X_all = X_train[idx_all]
            label_all = label_train[idx_all]
            num_class = len(np.unique(label_test))

            centr_model = GlobalModel(args, X_all, num_class).set_model()

            centr_model.fit(
                X_all, label_all, batch_size=args.batch_size, epochs=args.nround, verbose=0)
            centr = centr_model.evaluate(X_test, label_test)[1]
            acc_cntr[r, i] = centr

            if i == 0:
                # train classifier on data of one user
                ind = centr
            acc_ind[r, i] = ind

            start = time.time()

            fl_model = GlobalModel(args, X_all, num_class).set_model()

            for rr in range(args.nround):
                print(f'==== Round {rr+1} / {args.nround} =====')

                user_w_list, user_ndat_list = [], []
                for c in range(i+1):

                    # get a global model and set fedavg weight to it
                    local_model = GlobalModel(
                        args, X_all, num_class).set_model()
                    local_model.set_weights(
                        fl_model.get_weights())  # current_weight

                    local_model.fit(X_train[user_list[c]], label_train[user_list[c]], batch_size=args.batch_size,
                                    epochs=args.epoch, verbose=0) # , validation_data=(X_test, label_test)

                    # get each trained weight
                    user_w_list.append(local_model.get_weights())
                    user_ndat_list.append(len(X_train[user_list[c]]))

                # calcurate FedAvg
                new_w = fed_avg(user_w_list, user_ndat_list)
                # set new weight to a global model
                fl_model.set_weights(new_w)

            fed = fl_model.evaluate(X_test, label_test)[1]

            end = time.time()

            acc_fed[r, i] = fed
            time_fed[r, i] = end - start

    xval = np.arange(1, args.num_users+1)
    get_result(xval, acc_cntr, acc_ind, acc_fed, time_fed, args, method='fed', setting='users')
