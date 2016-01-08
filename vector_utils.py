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

def confusion_matrix(y_pred, y_true, labels=None):
    # Send only filtered y values.
    y_pred, y_true = np.array(y_pred).flatten().squeeze(), np.array(y_true).flatten().squeeze()
    if labels is None:
        labels = list(set(y_true).union(set(y_pred)))
    n_labels = len(labels)
    print "[%s] labels = %s" % (n_labels, labels) 
    CM = coo_matrix((np.ones_like(y_true, dtype="int"), (y_true, y_pred)), shape=(n_labels, n_labels)).todense()
    return CM

def get_prf_scores(cm):
    scores = dict()
    TP = np.diag(cm)
    FP = np.squeeze(np.asarray((np.sum(cm, axis=1)))) - TP
    FN = np.squeeze(np.asarray((np.sum(cm, axis=0)))) - TP
    scores["TP"] = TP
    scores["FP"] = FP
    scores["FN"] = FN
    precision = TP * 1. / (TP + FP)
    recall = TP * 1. / (TP + FN)
    f1_score = 2*precision*recall / (precision + recall)
    macro_f1 = np.mean(f1_score)
    scores["precision"] = precision
    scores["recall"] = recall
    scores["f1_score"] = f1_score
    scores["macro_f1"] = macro_f1
    micro_precision = np.sum(TP) * 1. / np.sum(TP + FP)
    micro_recall = np.sum(TP) * 1. / np.sum(TP + FN)
    micro_f1 = 2*micro_precision*micro_recall / (micro_precision+micro_recall)
    scores["micro_precision"] = micro_precision
    scores["micro_recall"] = micro_recall
    scores["micro_f1"] = micro_f1
    return scores


def get_eval_scores(y_pred, y_true, labels=None):
  return get_prf_scores(confusion_matrix(y_pred, y_true, labels=labels))
