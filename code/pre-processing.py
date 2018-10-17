# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 14:37:26 2018

@author: MortZ
"""

'''
load original data
* list of adjacency matrices of 100 molecules
* list of feature matrices of 100 molecules
* labels for graph classification: 500 labels for both classes
'''
# load aj_matrix, features matrix and labels
AM, FM, labels = load_tox21_SR_MMP()

# convert data to list instead of object
AM = AM.tolist()
FM = FM.tolist()

'''
reduce dataset from 5810 molecules (4892 class 1, 918 class 2) 
to 1000 molecules such that each class has 500 molecules.
'''
AM1000 = [ [] for x in range(1000)]
FM1000 = [ [] for x in range(1000)]
labels1000 = [ [] for x in range(1000)]
# class 1 = first 500 elements of AM
for i in range(500): # len(AM1000)/2
    AM1000[i] = AM[i]
    FM1000[i] = FM[i]
    labels1000[i] = labels[i]
    
# class 2 = from index 4892
for i in range(500):
    AM1000[i+500] = AM[i+4892]
    FM1000[i+500] = FM[i+4892]
    labels1000[i+500] = labels[i+4892]
    
AM = AM1000
FM = FM1000
labels = labels1000

labels = pd.DataFrame(labels)
labels = labels.astype('int32')
labels = labels.values

'''
Create a 'super node' connected to all other nodes in the graph which can then
be interpreted as the graph-level classification. 
Iterate through AM and FM, obtain number of nodes.
For AM, add 1 node connected to all others
For FM, add 1 node with feature values = 0
'''
AM_s = list.copy(AM)
FM_s = list.copy(FM)

n_rows = [ [] for x in range(len(AM))]
for i in range(len(AM)):
    n_rows[i] = len(AM[i])
    n_super_row = np.ones(n_rows[i])
    n_super_col = np.ones((n_rows[i]+1, 1))
    n_super_col[-1] = 0
    AM_s[i] = np.vstack((AM_s[i], n_super_row))
    AM_s[i] = np.hstack((AM_s[i], n_super_col))

n_super = np.zeros(75)
for i in range(len(FM)):
    FM_s[i] = np.vstack((FM_s[i], n_super))
    
n_rows_super = list.copy(n_rows)
for i in range(len(n_rows)):
    n_rows_super[i] =  n_rows_super[i]+1
    
    
'''
create list with index of super nodes when all graphs are concatenated
'''
idx_super = [ [] for x in range(len(n_rows_super))]
cum_sum = 0
for i in range(len(n_rows_super)):
    cum_sum = cum_sum + n_rows_super[i] 
    idx_super[i] = cum_sum
    

###############################################################################

'''
for index 0 to 99

training set class 1 = index 0 to 29  
training set class 2 = index 50 to 79    

testing set class 1 = index 30 to 49 
testing set class 2 = index 80 to 99
'''
tr1_lb = 0
tr1_ub = 300
tr2_lb = 500
tr2_ub = 800

tst1_lb = 300
tst1_ub = 400
tst2_lb = 800
tst2_ub = 900

val1_lb = 400
val1_ub = 500
val2_lb = 900
val2_ub = 1000

target0 = [0, 0]
target1 = [0, 1]
target2 = [1, 0]

#################################
'''
features = (1757, 75)
concatenation of all FMs
'''
features = FM_s[0]
for i in range(len(FM_s)-1):
    features = np.concatenate((features, FM_s[i+1]))
    
features = sp.lil_matrix(features)

'''
adj = (1757, 1757)
block diag matrix of all AM_s
'''
adj = []
for i in range(len(AM_s)):
    adj = sl.block_diag(adj, AM_s[i])
adj = np.delete(adj, (0), axis=0) # delete 1st row from initialising adj
adj = sp.csr_matrix(adj)

####################################

'''
y_train = labels for tr data (super nodes only)
'''
labels_tr = target0
for i in range(idx_super[-1]): # i=0:1757
    labels_tr = np.vstack((labels_tr, target0)) # make all nodes have no labels
labels_tr = np.delete(labels_tr, (0), axis=0) # delete first row from initialising
    
mask_tr = np.zeros((18797,1), dtype=bool)
    
for i in range(len(labels_tr)+1): 
    for j in range(len(idx_super)): # 100
        if i < idx_super[tr1_ub]: # tr1 = first 30 graphs
            if i == idx_super[j]:
                labels_tr[i-1] = target1 # label of super node = class 1
                mask_tr[i-1] = True
        if idx_super[tr2_lb] <= i < idx_super[tr2_ub]: # range if tr2 (864-1405)
            if i == idx_super[j]:
                labels_tr[i-1] = target2 # label of super node = class 2
                mask_tr[i-1] = True


'''
y_test = labels for tst data (super nodes only)
'''
labels_tst = target0
for i in range(idx_super[-1]): # i=0:1757
    labels_tst = np.vstack((labels_tst, target0)) # make all nodes have no labels
labels_tst = np.delete(labels_tst, (0), axis=0) # delete first row from initialising
    
mask_tst = np.zeros((18797,1), dtype=bool)
    
for i in range(len(labels_tst)+1): 
    for j in range(len(idx_super)): # 100
        if idx_super[tst1_lb] <= i < idx_super[tst1_ub]: # tst1 range
            if i == idx_super[j]:
                labels_tst[i-1] = target1 # label of super node = class 1
                mask_tst[i-1] = True
        if idx_super[tst2_lb] <= i < idx_super[tst2_ub]: # tst2 range
            if i == idx_super[j]:
                labels_tst[i-1] = target2 # label of super node = class 2
                mask_tst[i-1] = True


'''
y_val = labels for val data (super nodes only)
'''
labels_val = target0
for i in range(idx_super[-1]): # i=0:1757
    labels_val = np.vstack((labels_val, target0)) # make all nodes have no labels
labels_val = np.delete(labels_val, (0), axis=0) # delete first row from initialising
    
mask_val = np.zeros((18797,1), dtype=bool)
    
for i in range(len(labels_val)+1): 
    for j in range(len(idx_super)): # 100
        if idx_super[val1_lb] <= i < idx_super[val1_ub]: # val1 range
            if i == idx_super[j]:
                labels_val[i-1] = target1 # label of super node = class 1
                mask_val[i-1] = True
        if idx_super[val2_lb] <= i: # val2 range
            if i == idx_super[j]:
                labels_val[i-1] = target2 # label of super node = class 2
                mask_val[i-1] = True
                
                
np.save('adjFINAL1000.npy', adj)
np.save('featuresFINAL1000.npy', features)
np.save('idx_superFINAL1000.npy', idx_super)
np.save('y_trainFINAL1000.npy', y_train)
np.save('y_testFINAL1000.npy', y_test)
np.save('y_valFINAL1000.npy', y_val)
np.save('train_maskFINAL1000.npy', train_mask)
np.save('test_maskFINAL1000.npy', test_mask)
np.save('val_maskFINAL1000.npy', val_mask)