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

# graph
import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_palette('Set2')
sns.set_style('whitegrid')
sns.set_context(font_scale=1.3)


args = args_parser()
ir_method = 'PCA'           # method to make IR
ndat = args.ndat

if __name__ == '__main__':

    # nanc_list = [25, 50, 75, 100, 200, 500, 1000]
    nd_list = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99] # the ratio of variance to remain in PCA
    acc_cntr = np.zeros([args.repeat, args.num_users])
    acc_ind = np.zeros([args.repeat, args.num_users])
    acc_dc = np.zeros([args.repeat, len(nd_list), args.num_users])
    start_time= time.time()

    
    for r in range(args.repeat):
        print(f'Round {r+1}')
        random.seed(r)
        X_train, X_test, X_anc, label_train, label_test, user_list = get_dataset(args)
        # anc_list = [make_anchors(X_anc, nanc, args) for nanc in nd_list]
        anc = make_anchors(X_train, args.nanc, args)

        assert len(label_test) == args.ntest
        
        for ii in tqdm(range(args.num_users)):
            print(f'User {ii+1}: {Counter(label_train[user_list[ii]])}')

            # Main
            # train classifier on all data
            idx_all = np.ravel(user_list[:ii+1]).tolist()
            X_all = X_train[idx_all]
            label_all = label_train[idx_all]
            num_class = len(np.unique(label_test))

            centr_model = GlobalModel(args, X_all, num_class).set_model()
            
            if args.model == 'knn':
                centr_model.fit(X_all, label_all)
                centr = centr_model.score(X_test, label_test)
            else: # keras model
                centr_model.fit(X_all, label_all, batch_size=args.batch_size, epochs=args.epoch, verbose=0)
                centr = centr_model.evaluate(X_test, label_test)[1]
            
            acc_cntr[r, ii] = centr
            
            if ii == 0:
                # train classifier on data of one user
                ind = centr
            
            acc_ind[r, ii] = ind

            # Proposed method(User 1 has test data)
            # pseudo-split of data

            d_cr_list = []
            for n, value in enumerate(nd_list): # nd_list, nanc_list

                print(f'= {n+1}: {value} = ')

                Div_data = []
                # anc = make_anchors(X_train, value, args) # comment out when you explore d_ir
                
                for i in range(ii+1):

                    user_idx_i = user_list[i]
                    Div_data.append({'X':X_train[user_idx_i], 'Xtest':X_test, 'anc':anc}) 
              
                X_dc, X_test_dc, d_cr = data_collaboration(Div_data, ir_method, args, d_ir=value) # d_ir = value
                
                dc_model = GlobalModel(args, X_dc, num_class).set_model()
                d_cr_list.append(d_cr)

                if args.model == 'knn':
                    dc_model.fit(X_dc, label_all)
                    dc = dc_model.score(X_test_dc, label_test)
                
                else:
                    dc_model.fit(X_dc, label_all, batch_size=args.batch_size, epochs=args.epoch)
                    dc = dc_model.evaluate(X_test_dc, label_test)[1]
                
                acc_dc[r, n, ii] = dc
            
    end_time = time.time()

    print('Time for computation: ', end_time - start_time)
    centr = np.round(np.mean(acc_cntr, 0), decimals=3)
    ind = np.round(np.mean(acc_ind, 0), decimals=3)
    dc = np.round(np.mean(acc_dc, 0), decimals=3)

    print('Averaged over {} runs'.format(args.repeat))
    print('Centralized average accuracy:', centr)
    print('Individual average accuracy:', ind)
    print('Collaboration average accuracy:', dc)

    dir_path = "/Users/nedo_m02/Desktop/pytorch_practice/FL"
    xval = np.arange(1, args.num_users + 1)
    colorlist = np.linspace(0.9, 0.2, len(nd_list)).tolist()
    colorlist = [str(i) for i in colorlist]
    
    plt.figure(figsize=(13,5))
    plt.plot(xval, centr, label='Centralized', marker=".")
    plt.plot(xval, ind, label='Indivisual (User1)', marker=".")
    for n, value in enumerate(nd_list):
        plt.plot(xval, dc[n], label=f'{value} var ({d_cr_list[n]} dims)', marker=".", linestyle='--', color=colorlist[n]) # f'{value} dims'
    plt.xlabel('Number of users')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of {} ({})'.format(args.dataset.upper(), ['IID' if args.iid else 'Non-IID'][0]))
    plt.ylim(min(ind)-0.1, max(centr)+0.1)
    plt.legend()
    if args.save_fig:
        plt.savefig(dir_path + '/save/figures/dc_%s_%s_%sanc_%susers_iid[%s]_%srun.png'%(args.dataset, args.model, args.nanc, args.num_users, args.iid, args.repeat))
    else:
        pass
    plt.show()
