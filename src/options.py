import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # dc arguments
    parser.add_argument('--num_users', type=int, default=10,
                        help="number of users: K")
    parser.add_argument('--repeat', type=int, default=3,
                        help='number of repeat(epoch) of dc analysis')
    parser.add_argument('--ndat', type=int, default=100,
                        help="number of data per users: N")
    parser.add_argument('--ntest', type=int, default=1000,
                        help='number of test data')
    parser.add_argument('--anc_type', type=str, default='random',
                        choices=['random', 'gan', 'saved'], help="method to create anchor data")
    parser.add_argument('--nanc', type=int, default=500,
                        help="number of anchor data")
    parser.add_argument('--d_ir', type=int, default=50, help='dimension \
                        of intermediate representation')

    # data arguments
    parser.add_argument('--dataset', type=str, choices=['mnist', 'fashion_mnist', 'cifar'],
                        default='fashion_mnist', help="name of dataset")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')

    # model arguments
    parser.add_argument('--model', type=str, default='mlp',
                        choices=['mlp', 'cnn', 'knn'])
    parser.add_argument('--n_neighbors', type=int, default=6,
                        help='for LLE, LPP, KNN claffifier')
    parser.add_argument('--lr', type=int, default=0.05,
                        help='learning rate of decision tree based classifier')
    parser.add_argument('--max_depth', type=int, default=10,
                        help='max depth of decision tree based classifier')
    parser.add_argument('--epoch', type=int, default=3,
                        help='epochs of local training in federated learning')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size of local training in federated learning')
    parser.add_argument('--nround', type=int, default=24,
                        help='number of round for whole training of federated averaging')
    parser.add_argument('--optimizer', type=str, default='adamax',
                        choices=['sgd', 'adamax', 'adadelta', 'adam'], help='optimizer for training of neural network')

    # other
    parser.add_argument('--save_fig', type=int, default=1,
                        help='Default set to save plot. Set 0 not to save a figure.')
    # parser.add_argument('--dc_params', type=int, default='nanc', choices=['nanc', 'd_ir'], 
    #                     help='which parameters to explore. nanc_list=[25, 50, 75, 100, 200 , 500, 1000] \
    #                         nd_list = [int(ndat * rate) for rate in [0.1, 0.2, 0.35, 0.5, 0.75, 1]]')

    args = parser.parse_args()
    return args
