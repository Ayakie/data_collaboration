import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # dc arguments
    parser.add_argument('--num_users', type=int, default=5,
                        help="number of users: K")
    parser.add_argument('--repeat', type=int, default=10,
                        help='number of repeat(epoch) of dc analysis')
    parser.add_argument('--ndat', type=int, default=100,
                        help="number of data per users: N")
    parser.add_argument('--ntest', type=int, default=1000,
                        help='number of test data')
    parser.add_argument('--anc_type', type=str, default='random',
                        choices=['random', 'gan_new', 'gan', 'raw', 'augmented'], help="method to create anchor data. \
                            'gan_new' is to train GAN. 'gan' is to load pretrained GAN model. ")
    parser.add_argument('--nanc', type=int, default=500,
                        help="number of anchor data")
    parser.add_argument('--d_ir', type=int, default=50, help='dimension \
                        of intermediate representation')
    parser.add_argument('--ir_method', type=str, default='SVD', choices=[
                        'PCA', 'ICA', 'LPP', 'LLE', 'SVD'], help='method to make intermediate representation')

    # data arguments
    parser.add_argument('--dataset', type=str, choices=['mnist', 'fashion_mnist', 'cifar'],
                        default='fashion_mnist', help="name of dataset")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')

    # model arguments
    parser.add_argument('--model', type=str, default='mlp',
                        choices=['mlp', 'cnn', 'cnn1d', 'knn', 'svm'])
    parser.add_argument('--n_neighbors', type=int, default=6,
                        help='for LLE, LPP, KNN classifier')
    parser.add_argument('--epoch', type=int, default=1,
                        help='epochs of local training in federated learning')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size of local training in federated learning')
    parser.add_argument('--nround', type=int, default=24,
                        help='number of round for whole training of federated averaging')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['sgd', 'adamax', 'adadelta', 'adam'], help='optimizer for training of neural network')

    # other
    parser.add_argument('--save_fig', type=int, default=1,
                        help='Default set to save plot. Set 0 not to save a figure.')

    args = parser.parse_args()
    return args
