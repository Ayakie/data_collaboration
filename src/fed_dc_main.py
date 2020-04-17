import numpy as np
import random
from collections import Counter
import time
from tqdm import tqdm
from utils import get_dataset, fed_avg, exp_details, exp_results, get_3col_plt
from models import GlobalModel
from sampling import make_anchors
from options import args_parser
from dc_utils import data_collaboration
from keras.models import model_from_config

# graph
import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_palette('Set2', n_colors=5)
sns.set_style('darkgrid')
sns.set_context(font_scale=1.3)

args = args_parser()
ir_method = 'PCA'

if __name__ == '__main__':

    print('Number of User : ', args.num_users)

    centr_score, ind_score, dc_score, = [], [], []
    fl_score = np.zeros((args.repeat, 3, args.nround))

    for r in range(args.repeat):
        print('Repeat: ', r+1)
        random.seed(r)
        X_train, X_test, X_anc, label_train, label_test, user_list = get_dataset(args)

        # Main
        idx_all = np.ravel(user_list).tolist()
        X_all = X_train[idx_all]
        label_all = label_train[idx_all]
        num_class = len(np.unique(label_test))

        # data collaboration
        # pseudo-split of data
        Div_data = []
        anc = make_anchors(X_anc, args.nanc, args)
        
        for i in range(args.num_users):
            user_idx_i = user_list[i]
            Div_data.append(
                {'X': X_train[user_idx_i], 'Xtest': X_test, 'anc': anc})
        
        X_dc, X_test_dc, _ = data_collaboration(Div_data, ir_method, args, args.d_ir)

        # =========== Build Model =============

        centr_model = GlobalModel(args, X_all, num_class).set_model()
        ind_model = GlobalModel(args, X_all, num_class).set_model()
        dc_model = GlobalModel(args, X_dc, num_class).set_model()
        fl_model = GlobalModel(args, X_all, num_class).set_model()

        # ============ Training ============

        # whole training on all data
        print('----- Whole Training -----')
        centr_hist = centr_model.fit(
            X_all, label_all, batch_size=args.batch_size, epochs=args.nround, validation_data=(X_test, label_test), verbose=0)
        centr_score.append([centr_hist.history['loss'], centr_hist.history['val_loss'],
                            centr_hist.history['val_sparse_categorical_accuracy']])

        # local training on User 1
        print('------ Local Training -----')
        ind_hist = ind_model.fit(X_train[user_list[0]], label_train[user_list[0]],
                                          batch_size=args.batch_size, epochs=args.nround, validation_data=(X_test, label_test), verbose=0)
        ind_score.append([ind_hist.history['loss'], ind_hist.history['val_loss'],
                          ind_hist.history['val_sparse_categorical_accuracy']])

        # proposed method
        print('------ Data Collaboration -----')
        dc_hist = dc_model.fit(
            X_dc, label_all, batch_size=args.batch_size, epochs=args.nround, validation_data=(X_test_dc, label_test))
        dc_score.append([dc_hist.history['loss'], dc_hist.history['val_loss'],
                         dc_hist.history['val_sparse_categorical_accuracy']])

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

            fl_train_score = fl_model.evaluate(X_all, label_all)
            fl_test_score = fl_model.evaluate(X_test, label_test)
            
            fl_score[r, 0, rr] = fl_train_score[0]
            fl_score[r, 1, rr] = fl_test_score[0]
            fl_score[r, 2, rr] = fl_test_score[1]
            # fl_score.append([fl_train_score[0], fl_test_score[0], fl_test_score[1]])
    
            print(f'Accuracy of FL in round {rr+1}: {fl_test_score[1]}')

    centr_score = np.round(np.mean(centr_score, axis=0), decimals=3)
    ind_score = np.round(np.mean(ind_score, axis=0), decimals=3)
    dc_score = np.round(np.mean(dc_score, axis=0), decimals=3)
    fl_score = np.round(np.mean(fl_score, axis=0), decimals=3)

    exp_results(centr_score, ind_score, dc_score, fl_score, args)

    xlabel = 'Rounds for FL, Epochs for the other methods'
    xval = np.arange(1, args.nround + 1)
    get_3col_plt(centr_score, ind_score, dc_score, fl_score, args, xlabel, xval)