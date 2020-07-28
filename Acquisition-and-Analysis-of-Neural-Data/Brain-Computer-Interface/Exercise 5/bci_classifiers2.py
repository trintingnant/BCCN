import numpy as np

def cov_shrink(X):
    '''
    Estimate covariance of given data using shrinkage estimator.
    
    Synopsis:
        C= cov_shrink(X)
    Argument:
        X: data matrix (features x samples)
    Output:
        C: estimated covariance matrix
    '''
    Xc = X - np.mean(X, axis=1, keepdims=True)
    d, n = Xc.shape
    Cemp= Xc.dot(Xc.T)/(n-1)
    
    sumVarCij = 0
    for ii in range(d):
        for jj in range(d):
            varCij = np.var(Xc[ii,:]*Xc[jj,:])
            sumVarCij += varCij

    nu = np.mean(np.diag(Cemp))
    gamma = n/(n-1.)**2 * sumVarCij / sum(sum((Cemp-nu*np.eye(d,d))**2))
    S= gamma*nu*np.eye(d,d) + (1-gamma)*Cemp
    return S


def train_LDAshrink(X, y):
    '''
    Synopsis:
        w, b= train_LDAshrink(X, y)
    Arguments:
        X: data matrix (features X samples)
        y: labels with values 0 and 1 (1 x samples)
    Output:
        w: LDA weight vector
        b: bias term
    '''
    mu1 = np.mean(X[:, y==0], axis=1)
    mu2 = np.mean(X[:, y==1], axis=1)
    # pool centered features to estimate covariance on samples of both classes at once
    Xpool = np.concatenate((X[:, y==0]-mu1[:,np.newaxis], X[:, y==1]-mu2[:,np.newaxis]), axis=1)
    C = cov_shrink(Xpool)
    w = np.linalg.pinv(C).dot(mu2-mu1)
    b = w.T.dot((mu1 + mu2) / 2.)
    return w, b
