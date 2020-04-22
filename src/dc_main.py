import numpy as np
import random
from collections import Counter
import time
from tqdm import tqdm
from utils import get_dataset, get_result
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


args = args_parser()

if __name__ == '__main__':
    acc_cntr = np.zeros([args.repeat, args.num_users])
    acc_ind = np.zeros([args.repeat, args.num_users])
    acc_dc = np.zeros([args.repeat, args.num_users])

    time_dc = np.zeros([args.repeat, args.num_users])

    for r in range(args.repeat):
        print(f'Repeat: {r+1} / {args.repeat}')
        random.seed(r)
        X_train, X_test, X_anc, label_train, label_test, user_list = get_dataset(args)

        for i in tqdm(range(args.num_users)):
            print(f"User {i+1}: {Counter(label_train[user_list[i]])}")

            # Main
            # train classifier on all data
            idx_all = np.ravel(user_list[:i+1]).tolist()
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
            acc_cntr[r, i] = centr

            if i == 0:
                # train classifier on data of one user
                ind = centr
            acc_ind[r, i] = ind

            # Proposed method(User 1 has test data)
            # pseudo-split of data

            start = time.time()

            anc = make_anchors(X_anc)

            Div_data = []
            for ii in range(i+1):
                user_idx_i = user_list[ii]
                Div_data.append(
                    {'X': X_train[user_idx_i], 'Xtest': X_test, 'anc': anc})
            
            X_dc, X_test_dc, _ = data_collaboration(Div_data, args)

            assert len(X_dc) == len(X_all)

            dc_model = GlobalModel(args, X_dc, num_class).set_model()

            if args.model == 'knn' or args.model == 'svm':
                dc_model.fit(X_dc, label_all)
                dc = dc_model.score(X_test_dc, label_test)
            else:
                dc_model.fit(X_dc, label_all, batch_size=args.batch_size,
                             epochs=args.nround, verbose=0)
                dc = dc_model.evaluate(X_test_dc, label_test)[1]
            
            acc_dc[r, i] = dc

            end = time.time()
            time_dc[r, i] = end - start


    xval = np.arange(1, args.num_users+1)
    get_result(xval, acc_cntr, acc_ind, acc_dc, time_dc, args, method='dc', setting='users')
    
