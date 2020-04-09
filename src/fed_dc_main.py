import numpy as np
import random
from collections import Counter
import time
from tqdm import tqdm
from utils import get_dataset, fed_avg, exp_details
from models import GlobalModel
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

    fl_train_loss = np.zeros((args.repeat, args.nround))
    fl_test_loss = np.zeros((args.repeat, args.nround))
    fl_test_acc = np.zeros((args.repeat, args.nround))

    print('Number of User : ', args.num_users)

    centr_train_loss, centr_test_loss, centr_test_acc = [], [], []
    ind_train_loss, ind_test_loss, ind_test_acc = [], [], []
    dc_train_loss, dc_test_loss, dc_test_acc = [], [], []
    for r in range(args.repeat):
        print('Repeat: ', r+1)
        random.seed(args.seed)
        X_train, label_train, X_test, label_test, Xanc, user_list = get_dataset(args)

        # pseudo-split of data
        Div_data = []
        for i in range(args.num_users):
            user_idx_i = user_list[i]
            Div_data.append({'X': X_train[user_idx_i], 'Xtest': X_test, 'Xanc': Xanc,
                             'label_train': label_train[user_idx_i], 'label_test': label_test})

        # Main
        idx_all = np.ravel(user_list[:]).tolist()
        X_all = X_train[idx_all]
        label_all = label_train[idx_all]

        # data collaboration
        X_dc, X_test_dc = data_collaboration(Div_data, ir_method, args)

        # =========== Build Model =============
        input_shape = X_train.shape[1:]
        input_shape_dc = X_dc.shape[1:]
        num_class = len(np.unique(label_train))
        centr_model = GlobalModel(args, input_shape, num_class).set_model()
        ind_model = GlobalModel(args, input_shape, num_class).set_model()
        dc_model = GlobalModel(args, input_shape_dc, num_class).set_model()
        fl_model = GlobalModel(args, input_shape, num_class).set_model()

        # ============ Training ============

        # whole training on all data
        print('----- Whole Training -----')
        centr_hist_callback = centr_model.fit(
            X_all, label_all, batch_size=args.batch_size, epochs=args.nround, validation_data=(X_test, label_test))
        centr_train_loss.append(centr_hist_callback.history['loss'])
        centr_test_loss.append(centr_hist_callback.history['val_loss'])
        centr_test_acc.append(centr_hist_callback.history['val_sparse_categorical_accuracy'])

        # local training on User 1
        print('------ Local Training -----')
        ind_hist_callback = ind_model.fit(Div_data[0]['X'], Div_data[0]['label_train'],
                                          batch_size=args.batch_size, epochs=args.nround, validation_data=(X_test, label_test))
        ind_train_loss.append(ind_hist_callback.history['loss'])
        ind_test_loss.append(ind_hist_callback.history['val_loss'])
        ind_test_acc.append(ind_hist_callback.history['val_sparse_categorical_accuracy'])

        # proposed method
        print('------ Data Collaboration -----')
        dc_hist_callback = dc_model.fit(
            X_dc, label_all, batch_size=args.batch_size, epochs=args.nround, validation_data=(X_test_dc, label_test))
        dc_train_loss.append(dc_hist_callback.history['loss'])
        dc_test_loss.append(dc_hist_callback.history['val_loss'])
        dc_test_acc.append(dc_hist_callback.history['val_sparse_categorical_accuracy'])

        # federated learning
        print('----- Federated Learning -----')

        for rr in range(args.nround):
            print(f'==== Round {rr+1} / {args.nround} =====')

            user_w_list, user_ndat_list = [], []
            for user in range(args.num_users):
                
                # get a global model and set fedavg weight to it
                local_model = GlobalModel(args, input_shape, num_class).set_model()
                local_model.set_weights(fl_model.get_weights()) # current_weight

                local_model.fit(Div_data[user]['X'], Div_data[user]['label_train'], batch_size=args.batch_size,
                                epochs=args.epoch, validation_data=(X_test, label_test), verbose=0)

                # get each trained weight
                user_w_list.append(local_model.get_weights())
                user_ndat_list.append(len(Div_data[user]['X'])) # len(Div_data[user]['X'] == ndat

            # calcurate FedAvg
            new_w = fed_avg(user_w_list, user_ndat_list)
            # set new weight to a global model
            fl_model.set_weights(new_w)

            train_score = fl_model.evaluate(X_all, label_all)
            test_score = fl_model.evaluate(X_test, label_test)

            fl_train_loss[r, rr] = train_score[0]
            fl_test_loss[r, rr] = test_score[0]
            fl_test_acc[r, rr] = test_score[1]
            print(f'Accuracy of FL in round {rr+1}: {fl_test_acc}')
    
    centr_train_loss = np.round(np.mean(centr_train_loss, axis=0), decimals=3)
    centr_test_loss = np.round(np.mean(centr_test_loss, axis=0), decimals=3)
    centr_test_acc = np.round(np.mean(centr_test_acc, axis=0), decimals=3)

    ind_train_loss = np.round(np.mean(ind_train_loss, axis=0), decimals=3)
    ind_test_loss = np.round(np.mean(ind_test_loss, axis=0), decimals=3)
    ind_test_acc = np.round(np.mean(ind_test_acc, axis=0), decimals=3)

    dc_train_loss = np.round(np.mean(dc_train_loss, axis=0), decimals=3)
    dc_test_loss = np.round(np.mean(dc_test_loss, axis=0), decimals=3)
    dc_test_acc = np.round(np.mean(dc_test_acc, axis=0), decimals=3)

    fl_train_loss = np.round(np.mean(fl_train_loss, axis=0), decimals=3)
    fl_test_loss = np.round(np.mean(fl_test_loss, axis=0), decimals=3)
    fl_test_acc = np.round(np.mean(fl_test_acc, axis=0), decimals=3)

    print(f'Averaged over {args.repeat} runs')
    print('=== Centralized Model ===')
    print('Training loss: ', centr_train_loss)
    print('Test loss: ', centr_test_loss)
    print('Test accuracy: ', centr_test_acc)

    print('=== Individual Model ===')
    print('Training loss: ', ind_train_loss)
    print('Test loss: ', ind_test_loss)
    print('Test accuracy: ', ind_test_acc) 

    print('=== Data Collaboration ===')
    print('Training loss: ', dc_train_loss)
    print('Test loss: ', dc_test_loss)
    print('Test accuracy: ', dc_test_acc)  

    print('=== Federated Learning ===')
    print('Training loss of FL: ', fl_train_loss)
    print('Test loss of FL: ', fl_test_loss)
    print('Test accuracy of FL: ', fl_test_acc)

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    ax[0].plot(centr_train_loss, marker='.', label='centralized')
    ax[0].plot(ind_train_loss, linestyle='-.', marker='.', label='individual')
    ax[0].plot(dc_train_loss, marker='.', label='data collaboration')
    ax[0].plot(fl_train_loss, marker='.', label='federated learning')
    ax[0].set_xlabel('Rounds for FL, Epochs for the other methods')
    ax[0].legend()
    ax[0].set_title('Training Loss, NN')

    ax[1].plot(centr_test_loss, marker='.', label='centralized')
    ax[1].plot(ind_test_loss, linestyle='-.', marker='.', label='individual')
    ax[1].plot(dc_test_loss, marker='.', label='data collaboration')
    ax[1].plot(fl_test_loss, marker='.', label='federated learning')
    ax[1].set_xlabel('Rounds for FL, Epochs for the other methods')
    ax[1].legend()
    ax[1].set_title('Validation Loss, NN')

    ax[2].plot(centr_test_acc, marker='.', label='centralized')
    ax[2].plot(ind_test_acc, linestyle='-.', marker='.', label='individual')
    ax[2].plot(dc_test_acc, marker='.', label='data collaboration')
    ax[2].plot(fl_test_acc, marker='.', label='federated learning')
    ax[2].set_xlabel('Rounds for FL, Epochs for the other methods')
    ax[2].legend()
    ax[2].set_title('Validation Accuracy, NN')

    dir_path = "/Users/nedo_m02/Desktop/pytorch_practice/FL"
    if args.save_fig:
        plt.savefig(dir_path + '/save/figures/fed_dc_%s_%sanc_%susers_iid[%s]_%sround_%srun.png' 
            % (args.dataset, args.nanc, args.num_users, args.iid, args.nround, args.repeat))
    else:
        pass
    plt.show()
