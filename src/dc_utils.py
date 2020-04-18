import numpy as np
import scipy
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from sklearn.manifold import LocallyLinearEmbedding
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from packages.lpproj_LPP import LocalityPreservingProjection as LPP
from options import args_parser

args = args_parser()

def get_ir(X, Xtest, anc, method, args, d_ir):
    
    if method == 'PCA':
        f = PCA(d_ir, svd_solver='arpack')
    
    elif method == 'ICA':
        f = FastICA(d_ir)
        
    elif method == 'LPP':
        f = LPP(n_components=d_ir, n_neighbors=args.n_neighbors)
    
    elif method == 'LLE':
        f = LocallyLinearEmbedding(n_components=d_ir, n_neighbors=args.n_neighbors)

    elif method == 'SVD':
        # Either "arpack" for the ARPACK wrapper in SciPy(scipy.sparse.linalg.svds), or "randomized" for the randomized
        # algorithm due to Halko (2009).
        # f = TruncatedSVD(d_ir, algorithm='arpack')
        U1,s1,V1_T = scipy.linalg.svd(X, lapack_driver='gesvd')
        f = V1_T[:d_ir,:]
        X_tilde = np.dot(X, f.T)
        anc_tilde = np.dot(anc, f.T)
        Xtest_tilde = np.dot(Xtest, f.T)

        return X_tilde, Xtest_tilde, anc_tilde
        
    else:
        raise Exception('No method')
    
    f.fit(X)
    X_tilde = f.transform(X)
    Xtest_tilde = f.transform(Xtest)
    anc_tilde = f.transform(anc)
        
    return X_tilde, Xtest_tilde, anc_tilde


def get_cr(Div_tilde, d_cr):
    '''get collaboration representaion(CR)
    
    Parameters
    -----------
    Div_tilde: dict
        dict of each user's data after converting to IR
    d_cr: int
        usually it is eaqual to d_ir 
        (dimension of intermediate representation)
    Return
    ------
    X_hat_list: list of each user's CR
    Xtest_hat: CR of user1's test data
    
    '''
    
    anc_merged = np.hstack([i['anc_tilde'] for i in Div_tilde])
    
    # number of dimension of anchors per each user
    # anc_list_dims = [i['anc_tilde'].shape[1] for i in Div_tilde]
    # min_components = min(anc_list_dims)
    
    # check dimension
    # assert min_components >= d_cr, "n_components is too large, \
    #     maximum val is %s, but got %s" % (min_components, d_cr)
    
    U, s, V = scipy.linalg.svd(anc_merged, lapack_driver='gesvd')
    Z = U[:, :d_cr].T

    X_hat_list = []
    for i, user in enumerate(Div_tilde):
        
        # construct mapping function g
        g = np.dot(Z, np.linalg.pinv(user['anc_tilde'].T))
        X_hat = np.dot(user['X_tilde'], g.T)
        X_hat_list.append(X_hat)

        # first user has test data
        if i == 0:
            Xtest_hat = np.dot(user['Xtest_tilde'], g.T)
    
    return X_hat_list, Xtest_hat

# To do: enable to select different method per user
def data_collaboration(Div_data, method, args, d_ir=args.d_ir):
    '''compute whole process of DC
    
    Parameters
    ----------
    Div_data: list
        each element has a dict whose keys are 
        "X", "Xtest", "anc"
    method: str
        method of dimensionally reduction(PCA, LLE, LPP, SVD)
    d: dimension of intermediate representation
    d_cr: usually it is equal to d_ir
        dimention of left singular vector U
    
    '''
    
    # n_neighbors = args.n_neighbors

    Div_tilde = []
    for i, user in enumerate(Div_data):
        X_tilde, Xtest_tilde, anc_tilde = get_ir(user['X'], user['Xtest'], user['anc'], method, args, d_ir)
        Div_tilde.append({'X_tilde': X_tilde, 'Xtest_tilde': Xtest_tilde, 'anc_tilde': anc_tilde})
    
    anc_list_dims = [i['anc_tilde'].shape[1] for i in Div_tilde]
    d_cr = min(anc_list_dims)

    X_hat_list, Xtest_hat = get_cr(Div_tilde, d_cr=d_cr)
    X_hat_all = np.vstack(X_hat_list)
            
    return X_hat_all, Xtest_hat, d_cr