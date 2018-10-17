# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 21:18:13 2018

@author: MortZ

helpers
"""

import numpy as np
import scipy.sparse as sp
from gcn.utils import *
import pprint

'''
load data (adjacency matrices, feature matrices & labels)
for small tox21_100 dataset, single classification
returns data
'''
def load_100():
    adj_matrices = np.load('AM.npy')
    feature_matrices = np.load('features.npy')
    labels = np.load('labels.npy')
    return adj_matrices, feature_matrices, labels

def load_tox21_SR_MMP(): 
    adj_matrices = np.load('tox21_SR-MMP_AM.npy')
    feature_matrices = np.load('tox21_SR-MMP_features.npy')
    labels = np.load('tox21_SM-MMP_labels.npy')
    return adj_matrices, feature_matrices, labels

'''
argument: a list of adjacency matrices
returns: normalised adjacency matrices by adding identity matrix
'''
def list_norm_adj_matrix(adj_matrix):
    norm_adj_matrix = [ [] for x in range(len(adj_matrix))]
    for i in range(len(adj_matrix)):
        n = len(adj_matrix[i])
        norm_adj_matrix[i] = norm_adj_matrix[i] + np.identity(n)
    return norm_adj_matrix


'''
argument: adjacency matrix
returns: normalised adjacency matrix by adding identity matrix
'''
def norm_adj_matrix(adj_matrix):
    n = len(adj_matrix)
    norm_adj_matrix = adj_matrix + np.identity(n)
    return norm_adj_matrix


'''
argument: list of adjacency matrices
returns: list of corresponding degree matrices
'''
def list_adj_to_deg(adj_matrix):
    degree_matrices = [ [] for x in range(len(adj_matrix))]
    for x in range(len(adj_matrix)):
        n = len(adj_matrix[x])
        degree_matrices[x] = np.zeros((n,n))
        for i in range(n):
            sum_d = 0
            for j in range(n):
                sum_d += adj_matrix[x][i,j]
                degree_matrices[x][i,i] = sum_d
    return degree_matrices


'''
argument: adjacency matrix
returns: corresponding degree matrix
'''
def adj_to_deg(adj_matrix):
    n = len(adj_matrix)
    degree_matrix = np.zeros((n,n))
    for i in range(n):
        sum_d = 0
        for j in range(n):
            sum_d += adj_matrix[i,j]
            degree_matrix[i,i] = sum_d
    return degree_matrix


'''
argument: list of matrices 
returns: list of matrices of type csr_matrix
'''
def list_to_csr_matrix(matrices):
    csr_matrices = [ [] for x in range(len(matrices))]
    for i in range(len(matrices)):
        csr_matrices[i] = sp.csr_matrix(matrices[i])
    return csr_matrices


'''
argument: list of matrices
returns: concatenation (row wise) of all matrices as one matrix
'''
def concat_matrices(matrices):
    concatenation = matrices[0]
    for i in range(len(matrices)-1):
        concatenation = np.concatenate((concatenation, matrices[i+1]))
    return concatenation



'''
arguments: list of feature_matrices, lower bound index, upper bound index
returns: list of feature matrices within index range
'''
def feature_matrix_by_index(matrix_list, lb, ub):
    matrices_within_index = [ [] for x in range(ub-lb)]
    for i in range(lb, ub):
        matrices_within_index[i-lb] = matrix_list[i]
    return matrices_within_index


'''
arguments: labels set, index lower bound, index upper bound
returns: binary labels within index range
'''
def create_labels_range(labels, lb, ub):
    labels_range = np.zeros((ub-lb, 2), dtype='int64')
    for i in range(lb, ub):
        for j in range(0,2):
            labels_range[i-lb,j] = labels[i,j]
    return labels_range


'''
Flags can only be initialised once so this function must be called if the 
code is to be run more than once.
arguments: tf.flags.FLAGS
'''
def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)


'''
Prepare data for input to model
'''
def prepare_data(dataset_str, spFMtr, labels_tr_n, spFMtst, labels_tst_n, spFMall, labels_all_n, index_tst): #index_tst
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    #names  = [ 'x',      'y',        'tx',     'ty',       'allx',   'ally'    ]
    objects = [spFMtr, labels_tr_n, spFMtst, labels_tst_n, spFMall, labels_all_n]

    x, y, tx, ty, allx, ally = tuple(objects)
    test_idx_reorder = index_tst #parse_index_file("test_idx.txt")   #index_tst
    test_idx_range = np.sort(test_idx_reorder)


    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]


    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return features, y_train, y_val, y_test, train_mask, val_mask, test_mask
