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

from sklearn.decomposition._dict_learning import get_loss


def create_mnist_dataset(fraction_missing=0.1):

    from sklearn.datasets import load_digits

    X, _ = load_digits(return_X_y = True)

    missing_raw_values = np.random.uniform(0, 1, X.shape)
    missing_mask = missing_raw_values < fraction_missing
    missing_mask[1,1] = 1

    X_incomplete = X.copy()
    # fill missing entries with NaN
    X_incomplete[missing_mask] = np.nan

    return X, X_incomplete, missing_mask


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

def bench_feat(n_samples = 2000, n_features = 200, rank = 10, n_components = 10,
               alpha = 1):
    
    
    g_loss_na = []
    g_loss_sk = []
    g_time_na = []
    g_time_sk = []
    
    l_samples = [100, 500, 1000, 2500, 5000, 7500]
    l_features = [100, 250, 500, 1000, 2500, 5000, 7500, 10000]
#     for n_samples in l_samples:
    for n_features in l_features:
        print(n_features, end =';')
        loss_na = []
        loss_sk = []
        time_na = []
        time_sk = []
        for seed in range(4):
            print(seed, end ='.')
            X, X_na, missing_mask = create_rank_k_dataset(n_samples=n_samples, n_features=n_features, rank=rank, random_seed=seed)

            to = time.time()
            code_na, dict_na, _ = dict_learning_na(X, n_components = n_components,
                                                    alpha=alpha, ro = 2, T = 100)
            time_na.append(time.time() -to)
            loss_na.append(get_loss(X[300:], code_na[300:], dict_na, alpha))
            
            to = time.time()
            code_sk, dict_sk, _ = dict_learning_online(X, n_components = n_components,
                                                    alpha=alpha, batch_size=3, n_iter=100)
            time_sk.append(time.time() -to)
            loss_sk.append(get_loss(X[300:], code_sk[300:], dict_sk, alpha))
        
        g_loss_na.append(loss_na)
        g_loss_sk.append(loss_sk)
        g_time_na.append(time_na)
        g_time_sk.append(time_sk)
    
    
    g_loss_na = np.array(g_loss_na)
    g_loss_sk = np.array(g_loss_sk)
    g_time_na = np.array(g_time_na)
    g_time_sk = np.array(g_time_sk)

    plt.errorbar(l_features, np.mean(g_loss_na, axis=1), yerr= np.std(g_loss_na, axis=1), label = 'na')
    plt.errorbar(l_features, np.mean(g_loss_sk, axis=1), yerr= np.std(g_loss_sk, axis=1), label = 'sk')
    plt.xlabel('n_features')
    plt.ylabel('loss')
    plt.title(f'n_samples={n_samples}, rank={rank}, n_comp={n_components}')
    plt.legend()
    
    plt.figure()
    plt.errorbar(l_features, np.mean(g_time_na, axis=1), yerr= np.std(g_time_na, axis=1), label = 'na')
    plt.errorbar(l_features, np.mean(g_time_sk, axis=1), yerr= np.std(g_time_sk, axis=1), label = 'sk')
    plt.xlabel('n_features')
    plt.ylabel('time (s)')
    plt.title(f'n_samples={n_samples}, rank={rank}, n_comp={n_components}')
    plt.legend()


def bench_time_loss(n_samples = 6000, n_features = 2000, rank = 10, n_components = 10, alpha=1):

    X, _, _ = create_rank_k_dataset(n_samples=n_samples, n_features=n_features, rank=rank)

    l_time_sk = []
    l_loss_sk = []
    l_time_na = []
    l_loss_na = []
    for T in np.arange(5, 500, 40):
        print(T, end=';')
        to = time.time()
        code, dict_, _ = dict_learning_na(X, n_components = n_components,
                                            alpha=alpha, ro = 2, T = T)
        l_time_na.append(time.time() - to)
        l_loss_na.append(get_loss(X, code, dict_, alpha))

        to = time.time()
        code, dict_, _ = dict_learning_online(X, n_components = n_components,
                                            alpha=alpha, batch_size=1, n_iter=T)
        l_time_sk.append(time.time() - to)
        l_loss_sk.append(get_loss(X, code, dict_, alpha))

    plt.plot(l_time_na, l_loss_na, '*', label = 'na')
    plt.plot(l_time_sk, l_loss_sk, '*', label = 'sk')
    plt.legend()
    plt.ylabel('loss')
    plt.xlabel('time (s)')
    plt.title(f'n_samples={n_samples}, n_feat={n_features}, rank={rank}, n_comp={n_components}')



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

