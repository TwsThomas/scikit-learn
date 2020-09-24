import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy import linalg
import time


from sklearn.decomposition._dict_learning_na import sparse_encode_na,\
                                            update1, dict_learning_na,\
                                            update_dict_na, get_code_dict_learning_na,\
                                                reconstruction_error
                                            
from sklearn.decomposition.bench_dict_learning import create_rank_k_dataset, create_mnist_dataset

from fancyimpute import MatrixFactorization
from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, BiScaler

def bench_reconstruction(data = 'generated', n_samples = 4000, n_features = 200, rank = 10):
    
    T = 300
    if data == 'generated':
        l_components = [8,10,12]
        plt.title('generated data X = U*V (rank=10)')
        X, X_na, missing_mask = create_rank_k_dataset(n_samples=n_samples, n_features=n_features, rank=rank)
    elif data == 'mnist':
        l_components = [5,10,15]
        plt.title('Mnist data (1800 * 64)')
        X, X_na, missing_mask = create_mnist_dataset(fraction_missing=0.1)
    else:
        raise

    l_mse_dict_na = []
    l_mse_dict_na0 = []
    l_mse_MF = []
    l_mse_knn = []
    l_mse_si = []
    
    for ii, n_components in enumerate(l_components):
        print(n_components, end = ';')
        # dict_learning_na alpha=1
        _, dict_na, _ = dict_learning_na(X_na, n_components = n_components,
                                                        alpha=1, ro = 2, T = T)
        code_na = get_code_dict_learning_na(X, dict_na, alpha=1)
        mse_dict_na = reconstruction_error(X, np.dot(code_na, dict_na), missing_mask, verbose = 0)[0]
        l_mse_dict_na.append(mse_dict_na)
        
        # dict_learning_na alpha=0
        _, dict_na, _ = dict_learning_na(X_na, n_components = n_components,
                                                        alpha=0, ro = 2, T = T)
        code_na = get_code_dict_learning_na(X, dict_na, alpha=0)
        mse_dict_na = reconstruction_error(X, np.dot(code_na, dict_na), missing_mask, verbose = 0)[0]
        l_mse_dict_na0.append(mse_dict_na)
        
        # MF
        X_completed_test = MatrixFactorization(rank=n_components, epochs=5000, l2_penalty=1e-5, verbose = 0).fit_transform(X_na)
        mse_mf = reconstruction_error(X, X_completed_test, missing_mask, verbose = 0)[0]
        l_mse_MF.append(mse_mf)
        
        # KNN
        if ii == 0:
            X_completed_test = KNN(k=3, verbose = 0).fit_transform(X_na)
            mse_knn = reconstruction_error(X, X_completed_test, missing_mask, verbose = 0)[0]
        l_mse_knn.append(mse_knn)
        
        # SoftImpute
        if ii == 0:
            X_incomplete_normalized = BiScaler(verbose = 0).fit_transform(X_na)
            X_completed_test = SoftImpute(verbose = 0).fit_transform(X_incomplete_normalized)
            mse_si = reconstruction_error(X, X_completed_test, missing_mask, verbose = 0)[0]
        l_mse_si.append(mse_si)
        
    plt.plot(l_components, l_mse_dict_na, label = 'dict_na alpha=1')
    plt.plot(l_components, l_mse_dict_na0, label = 'dict_na alpha=0')
    
    plt.plot(l_components, l_mse_MF, label = 'MF')
    plt.plot(l_components, l_mse_knn, label = 'knn')
    plt.plot(l_components, l_mse_si, label = 'soft impute')
    
    plt.ylabel('MSE normalized')
    plt.xlabel('n_components')
    plt.legend()