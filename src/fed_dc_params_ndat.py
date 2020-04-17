import numpy as np
import random
from collections import Counter
import time
from tqdm import tqdm
from utils import get_dataset, fed_avg, exp_details, exp_results, get_3col_plt_2dc
from models import GlobalModel
from sampling import make_anchors
from options import args_parser
from dc_utils import data_collaboration
from keras.models import model_from_config
import time

# graph
import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_palette('Set2', n_colors=5)
sns.set_style('darkgrid')
sns.set_context(font_scale=1.3)

args = args_parser()
ir_method = 'PCA'

ndat_list = [51, 128, 256, 512, 750, 950, 1150, 1350, 1550, 1750]

if __name__ == '__main__':
    
    fl_score = np.zeros((args.repeat, 3, len(ndat_list)))
    centr_score = np.zeros((args.repeat, 3, len(ndat_list)))
    ind_score = np.zeros((args.repeat, 3, len(ndat_list)))
    dc_score = np.zeros((args.repeat, 2, 3, len(ndat_list)))

    start_time = time.time()
    
    for r in range(args.repeat):
        print('Repeat: ', r+1)
        random.seed(r)

        for n, ndat in enumerate(ndat_list):

            print('Number of data per user: ', ndat)
            X_train, X_test, X_anc, label_train, label_test, user_list = get_dataset(args, ndat=ndat)
            
            idx_all = np.ravel(user_list).tolist()
            X_all = X_train[idx_all]
            label_all = label_train[idx_all]
            num_class = len(np.unique(label_test))

            # data collaboration
            # pseudo-split of data
            Div_data_gan, Div_data_random = [], []
            anc_gan = make_anchors(X_anc, anc_type=args.anc_type)
            anc_random = make_anchors(X_anc, anc_type='random')

            for i in range(args.num_users):
                user_idx_i = user_list[i]
                Div_data_gan.append({'X': X_train[user_idx_i], 'Xtest': X_test, 'anc': anc_gan})
                Div_data_random.append({'X': X_train[user_idx_i], 'Xtest': X_test, 'anc': anc_random})
            
            X_dc1, X_test_dc1, _  = data_collaboration(Div_data_gan, ir_method, args) # gan
            X_dc2, X_test_dc2, _  = data_collaboration(Div_data_random, ir_method, args) # random

            # =========== Build Model =============

            centr_model = GlobalModel(args, X_all, num_class).set_model()
            ind_model = GlobalModel(args, X_all, num_class).set_model()
            dc_model1 = GlobalModel(args, X_dc1, num_class).set_model()
            dc_model2 = GlobalModel(args, X_dc2, num_class).set_model()
            fl_model = GlobalModel(args, X_all, num_class).set_model()

            # ============ Training ============

             # whole training on all data
            print('----- Whole Training -----')
            centr_hist = centr_model.fit(X_all, label_all, batch_size=args.batch_size, epochs=args.nround, validation_data=(X_test, label_test), verbose=0)
            
            # evaluate at the last epoch
            centr_score[r, 0, n] = centr_hist.history['loss'][-1]
            centr_score[r, 1, n] = centr_hist.history['val_loss'][-1]
            centr_score[r, 2, n] = centr_hist.history['val_sparse_categorical_accuracy'][-1]
            
            # local training on User 1
            print('------ Local Training -----')
            ind_hist = ind_model.fit(X_train[user_list[0]], label_train[user_list[0]],
                                          batch_size=args.batch_size, epochs=args.nround, validation_data=(X_test, label_test), verbose=0)
            
            ind_score[r, 0, n] = ind_hist.history['loss'][-1]
            ind_score[r, 1, n] = ind_hist.history['val_loss'][-1]
            ind_score[r, 2, n] = ind_hist.history['val_sparse_categorical_accuracy'][-1]

            # proposed method
            print('------ Data Collaboration -----')
            dc_hist1 = dc_model1.fit(X_dc1, label_all, batch_size=args.batch_size, epochs=args.nround, validation_data=(X_test_dc1, label_test))
            dc_hist2 = dc_model2.fit(X_dc2, label_all, batch_size=args.batch_size, epochs=args.nround, validation_data=(X_test_dc2, label_test))

            # --- gan ---
            dc_score[r, 0, 0, n] = dc_hist1.history['loss'][-1]
            dc_score[r, 0, 1, n] = dc_hist1.history['val_loss'][-1]
            dc_score[r, 0, 2, n] = dc_hist1.history['val_sparse_categorical_accuracy'][-1]

            # --- random ---
            dc_score[r, 1, 0, n] = dc_hist2.history['loss'][-1]
            dc_score[r, 1, 1, n] = dc_hist2.history['val_loss'][-1]
            dc_score[r, 1, 2, n] = dc_hist2.history['val_sparse_categorical_accuracy'][-1]           


            # federated learning
            print('----- Federated Learning -----')

            # fl_score_temp = []
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

                # calculate FedAvg
                new_w = fed_avg(user_w_list, user_ndat_list)
                # set new weight to a global model
                fl_model.set_weights(new_w)

                fl_train_score = fl_model.evaluate(X_all, label_all)
                fl_test_score = fl_model.evaluate(X_test, label_test)
        
                print(f'Accuracy of FL in round {rr+1}: {fl_test_score[1]}')
                
                # record federated learning accuracy at the last round
                if rr == args.nround - 1:
                    fl_score[r, 0, n] = fl_train_score[0]
                    fl_score[r, 1, n] = fl_test_score[0]
                    fl_score[r, 2, n] = fl_test_score[1]

    
    end_time = time.time()
    print('Time for comutation: ', start_time - end_time)

    centr_score = np.round(np.mean(centr_score, axis=0), decimals=3)
    ind_score = np.round(np.mean(ind_score, axis=0), decimals=3)
    dc_score = np.round(np.mean(dc_score, axis=0), decimals=3)
    fl_score = np.round(np.mean(fl_score, axis=0), decimals=3)
    
    exp_results(centr_score, ind_score, dc_score, fl_score, args)

    xlabel = f'Number of data per user ({args.num_users} users)'
    get_3col_plt_2dc(centr_score, ind_score, dc_score, fl_score, args, xlabel, ndat_list)


            