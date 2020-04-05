import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    # dc arguments
    parser.add_argument('--num_users', type=int, default=10,
                        help="number of users: K")
    parser.add_argument('--repeat', type=int, default=5,
                        help='number of repeat(epoch) of dc analysis')
    parser.add_argument('--ndat', type=int, default=100,
                        help="number of data per users: N")
    parser.add_argument('--anc_type', type=str, default='random', 
                        choices=['random', 'gan'], help="method to create anchor data")
    parser.add_argument('--nanc', type=int, default=2000, 
                        help="number of anchor data")
    parser.add_argument('--n_neighbors', type=int, default=6,
                        help='for LLE, LPP')
    parser.add_argument('--d', type=int, default=75, help='dimension\
                        of intermediate representation')

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    args = parser.parse_args()
    return args
