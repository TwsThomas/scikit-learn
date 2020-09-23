import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy import linalg
import time

from sklearn.decomposition import dict_learning_online
from sklearn.decomposition import sparse_encode

from sklearn.decomposition._dict_learning_na import sparse_encode_na,\
                                            update1, dict_learning_na,\
                                            update_dict_na




def create_rank_k_dataset(
                        n_samples=500,
                        n_features=200,
                        rank=4,
                        fraction_missing=0.1,
                        symmetric=False,
                        random_seed=0):

    np.random.seed(random_seed)
    U = np.random.randn(n_samples, rank)
    V = np.random.randn(rank, n_features)

    X = np.dot(U, V)

    missing_raw_values = np.random.uniform(0, 1, (n_samples, n_features))
    missing_mask = missing_raw_values < fraction_missing
    missing_mask[1,1] = 1

    X_incomplete = X.copy()
    # fill missing entries with NaN
    X_incomplete[missing_mask] = np.nan

    return X, X_incomplete, missing_mask

def bench_loss(n_samples = 300, n_features = 200, rank = 4, n_components = 4,
               alpha = 1, bench_time = False):

    X, X_na, missing_mask = create_rank_k_dataset(n_samples=n_samples, n_features=n_features, rank=rank)


    _, _, time_loss = dict_learning_na(X, n_components = n_components,
                                            alpha=alpha, ro = 2, T = 300)

    _, _, time_loss_sklearn = dict_learning_online(X, n_components = n_components,
                                            alpha=alpha, batch_size=1, n_iter=100)


    if bench_time:
        plt.plot([x[0] for x in time_loss], [x[1] for x in time_loss], label = 'thom')
        plt.plot([x[0] for x in time_loss_sklearn], [x[1] for x in time_loss_sklearn], '*', label = 'sklearn', ms = 7)
        plt.xlabel('time (s)')
    else:
        plt.plot([x[1] for x in time_loss], label = 'thom')
        plt.plot(np.array(range(len(time_loss_sklearn)))*3, [x[1] for x in time_loss_sklearn], '*', label = 'sklearn', ms = 7)
        plt.xlabel('n_iter')

    plt.legend()
    plt.title(f'n_samples={n_samples}, n_feat={n_features}, rank={rank}, n_comp={n_components}')
    plt.ylabel('loss')
    
    return time_loss, time_loss_sklearn



n_samples, n_features = 50, 25
rank = 12
np.random.seed(42)
U = np.random.randn(n_samples, rank)
V = np.random.randn(rank, n_features)
X = np.dot(U, V)

X_na = X.copy()
X_na[0,0] = np.nan
X_na[1,1] = np.nan

n_components = 11
code, dict_, loss_sklearn = dict_learning_online(X, n_components = n_components,
                                   alpha = .0001)

# def bench_loss(n_samples = 300, n_features = 200,
#                rank = 4, n_components = 4, alpha = 1, 
#                n_samples_test = 1000, bench_time = True):


#     U = np.random.randn(n_samples + n_samples_test, rank)
#     V = np.random.randn(rank, n_features)
#     X = np.dot(U, V)

#     X, X_test = X[:n_samples], X[n_samples:]
#     X_na = X.copy()
#     X_na[0,0] = np.nan
#     X_na[1,1] = np.nan

#     all_loss = []
#     for ro in [2]:
#         _, _, time_loss, loss_test = dict_learning_na(X, n_components = n_components,
#                                             alpha=alpha, ro = ro, X_test = X_test, T = 300)
#         if bench_time:
#             plt.plot([x[0] for x in time_loss], [x[1] for x in time_loss], label = 'thom')
#         else:
#             plt.plot([x[1] for x in time_loss], label = 'thom')
#         plt.plot(loss_test, '--', label = '' + ' test')
#         all_loss.append(time_loss)

#     _, _, time_loss_sklearn, loss_sklearn_test = dict_learning_online(X, n_components = n_components,
#                                             alpha=alpha, batch_size=1, n_iter=300, X_test = X_test)
#     all_loss.append(time_loss_sklearn)
    
#     if bench_time:
#         plt.plot([x[0] for x in time_loss_sklearn], [x[1] for x in time_loss_sklearn], '*', label = 'sklearn', ms = 7)
#     else:
#         plt.plot([x[1] for x in time_loss_sklearn], '*', label = 'sklearn', ms = 7)
#     plt.plot(loss_sklearn_test, '--', label = 'sklearn test')
#     plt.legend()
#     plt.title(f'n_samples={n_samples}, n_feat={n_features}, rank={rank}, n_comp={n_components}')
#     plt.ylabel('loss')
#     plt.xlabel('time (s)')
#     # plt.ylim((min(loss_sklearn)*.8, max(loss_sklearn)*1.3))
#     # plt.yscale('log')
#     # plt.xscale('log')
#     return all_loss

def bench_dict_learning():
    
    def get_time_dict(n_samples = 100, n_features=50, n_iter = 10):
        X = np.random.randn(n_samples, n_features)
        n_components = 10

        to = time.time()
        for _ in range(n_iter):
            code, D, loss_sk = dict_learning_online(X, n_components = n_components)
        t1 = time.time()
        print('dict_learning took:\t', (t1 - to)/n_iter, 'seconds')
        t = (t1 - to)/n_iter

        to = time.time()
        for _ in range(n_iter):
            code_na, D, loss = dict_learning_na(X, n_components = n_components)
        t1 = time.time()
        print('dict_learning_na took:\t', (t1 - to)/n_iter, 'seconds')
        t_na = (t1 - to)/n_iter

        return t, t_na

    ln = [50, 100, 1000, 10000]
    lt, lt_na = [], []
    for n_samples in ln:
        t, t_na = get_time_dict(n_samples = n_samples, n_features=200, n_iter=1)
        lt.append(t)
        lt_na.append(t_na)

    plt.plot(ln, lt, label = 'sklearn')
    plt.plot(ln, lt_na, label = 'na')
    plt.legend()
    plt.xscale('log')
    plt.title('dict_learning_online')
    plt.ylabel('time (s)')


def bench_sparse_encode(n_samples = 100, n_features=50, n_iter = 10):
    
    def get_time_sparse():
        X = np.random.randn(n_samples, n_features)
        code, dict_, loss_sk = dict_learning_online(X, n_components = 12, alpha = 1)

        to = time.time()
        for _ in range(n_iter):
            sparse_encode(X, dict_, alpha=1)
        t1 = time.time()
        print('sparse encode took:\t', (t1 - to)/n_iter, 'seconds')
        t = (t1 - to)/n_iter

        to = time.time()
        for _ in range(n_iter):
            sparse_encode_na(X, dict_, alpha=1)
        t1 = time.time()
        print('sparse encode_na took:\t', (t1 - to)/n_iter, 'seconds')
        t_na = (t1 - to)/n_iter

        return t, t_na

    ln = [10, 100,1000, 10000]
    lt, lt_na = [], []
    for n_samples in ln:
        t, t_na = get_time_sparse()
        lt.append(t)
        lt_na.append(t_na)

    plt.plot(ln, lt, label = 'sklearn')
    plt.plot(ln, lt_na, label = 'na')
    plt.legend()
    plt.xscale('log')
    plt.title('sparse encode')
    plt.ylabel('time (s)')

if __name__ == '__main__':
    bench_sparse_encode()
    plt.figure()
    bench_dict_learning()
    plt.figure()
    bench_loss()