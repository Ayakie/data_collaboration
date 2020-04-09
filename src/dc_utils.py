import numpy as np
import scipy
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from sklearn.manifold import LocallyLinearEmbedding
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from packages.lpproj_LPP import LocalityPreservingProjection as LPP

def get_ir(X, Xtest, Xanc, d_ir, method, n_neighbors):
    
    if method == 'PCA':
        f = PCA(d_ir, svd_solver='full')
    
    elif method == 'ICA':
        f = FastICA(d_ir)
        
    elif method == 'LPP':
        f = LPP(n_components=d_ir, n_neighbors=n_neighbors)
    
    elif method == 'LLE':
        f = LocallyLinearEmbedding(n_components=d_ir, n_neighbors=n_neighbors)

    else:
        # Either "arpack" for the ARPACK wrapper in SciPy
        # (scipy.sparse.linalg.svds), or "randomized" for the randomized
        # algorithm due to Halko (2009).
        f = TruncatedSVD(d_ir, algorithm='arpack')
#         U1,s1,V1_T = sc.linalg.svd(X, lapack_driver='gesvd')
#         f = V1_T[:d,:]
#         X_tilde = np.dot(X, f.T)
#         Xanc_tilde = np.dot(l, f.T)
#         Xtest_tilde = np.dot(X_test, f.T)
    
    f.fit(X)
    X_tilde = f.transform(X)
    Xtest_tilde = f.transform(Xtest)
    Xanc_tilde = f.transform(Xanc)
        
    return X_tilde, Xtest_tilde, Xanc_tilde


def get_cr(Div_tilde, d_cr):
    '''get collaboration representaion(CR)
    
    Parameters
    -----------
    Div_tilde: dict
        dict of each user's data after converting to IR

    Return
    ------
    X_hat_list: list of each user's CR
    Xtest_hat: CR of user1's test data
    
    '''
    
    anc_merged = np.hstack([i['Xanc_tilde'] for i in Div_tilde])
    
    # number of dimension of anchors per each user
    anc_list_dims = [i['Xanc_tilde'].shape[1] for i in Div_tilde]
    min_components = min(anc_list_dims)
    
    # check dimension
    assert min_components >= d_cr, "n_components is too large, \
        maximum val is %s, but got %s" % (min_components, d_cr)

    X_hat_list = []
    for i, user in enumerate(Div_tilde):
        U, s, V = scipy.linalg.svd(anc_merged, lapack_driver='gesvd')
        Z_T = U[:, :d_cr].T
        # construct mapping function g
        g = np.dot(Z_T, np.linalg.pinv(user['Xanc_tilde']).T)
        X_hat = np.dot(user['X_tilde'], g.T)
        X_hat_list.append(X_hat)

        # first user has test data
        if i == 0:
            Xtest_hat = np.dot(user['Xtest_tilde'], g.T)
    
    return X_hat_list, Xtest_hat

# To do: enable to select different method per user
def data_collaboration(Div_data, method, args):
    '''compute whole process of DC
    
    Parameters
    ----------
    Div_data: list
        each element has a dict whose keys are 
        "X", "Xtest", "Xanc", "label_train", "label_test"
    method: str
        method of dimensionally reduction(PCA, LLE, LPP, SVD)
    d: dimension of intermediate representation
    d_cr: usually it is equal to d
        dimention of left singular vector U
    
    '''
    d_ir = args.d_ir
    d_cr = d_ir
    n_neighbors = args.n_neighbors

    Div_tilde = []
    for i, user in enumerate(Div_data):
        X_tilde, Xtest_tilde, Xanc_tilde = get_ir(user['X'], user['Xtest'], user['Xanc'], d_ir, method, n_neighbors)
        Div_tilde.append({'X_tilde': X_tilde, 'Xtest_tilde': Xtest_tilde, 'Xanc_tilde': Xanc_tilde})
    
    X_hat_list, Xtest_hat = get_cr(Div_tilde, d_cr)
    X_hat_all = np.vstack(X_hat_list)
            
    return X_hat_all, Xtest_hat
        