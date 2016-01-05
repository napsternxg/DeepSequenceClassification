import numpy as np
from scipy.sparse import csr_matrix, coo_matrix

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

def save_arr_assparse(filename, array):
    return save_sparse_csr(filename, csr_matrix(array))

def load_sparse_asarr(filename):
    return load_sparse_csr(filename).toarray()


def to_onehot(Y, vector_size):
    Y_vec = []
    # Now y should be converted to one hot vector for each index value
    for i in xrange(len(Y)):
        Y_vec.append([])
        for j in xrange(len(Y[0])):
            y_vec = np.zeros(vector_size)
            y_vec[Y[i][j]] = 1
            Y_vec[-1].append(y_vec)
    return np.array(Y_vec, dtype='int32')

def to_onehot_char(X, vector_size):
    X_vec = np.zeros(X.shape + (vector_size,))
    for i in xrange(X_vec.shape[0]):
      for j in xrange(X_vec.shape[1]):
        for k in xrange(X_vec.shape[2]):
            try:
                X_vec[i,j,k,X[i,j,k]] = 1
            except:
                print X_vec.shape, X.shape, (i, j, k), X[i,j,k]
                raise Exception
    return X_vec

def onehot_to_idxarr(Y):
    return Y.argmax(axis=len(Y.shape) - 1)

def confusion_matrix(y_pred, y_true):
    n_labels = len(set(y_pred.tolist()).union(set(y_true.tolist())))
    CM = coo_matrix((np.ones(y_true.shape[0], dtype=np.int),(y_true, y_pred)), shape=(n_labels, n_labels)).toarray()
    return CM
