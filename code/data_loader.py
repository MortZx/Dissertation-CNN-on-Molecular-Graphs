# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 23:36:24 2018

@author: MortZ
"""

import numpy as np 
import scipy.sparse as sp

def load_data():
    adj = np.load('data/adj1000.npy').tolist()
    features = sp.lil_matrix(np.load('data/features1000.npy'))
    idx_super = np.load('data/idx_super1000.npy')
    y_train = np.load('data/y_train1000.npy')
    y_test = np.load('data/y_test1000.npy')
    y_val = np.load('data/y_val1000.npy')
    train_mask = np.load('data/train_mask1000.npy')
    test_mask = np.load('data/test_mask1000.npy')
    val_mask = np.load('data/val_mask1000.npy')
    return adj, features, idx_super, y_train, y_test, y_val, train_mask, test_mask, test_mask, val_mask

