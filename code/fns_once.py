# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 14:36:12 2018

@author: MortZ

Functions used once but kept for documentation purposes
"""

'''
once n1
'''
 # store number of rows of each graph in a list
 n_col = 75
 n_rows = [ [] for x in range(len(AM))]
 for i in range(len(AM)):
     n_rows[i] = len(AM[i])


'''
once n2
'''
n_tr1 = 0
n_tr2 = 0
n_tst1 = 0
n_tst2 = 0
n_all1 = 0
n_all2 = 0

# number of nodes in tr set class 1
for i in range(len(FMtr1)):
    for j in range(len(FMtr1[i])):
        n_tr1 += 1                              # = 513
# number of nodes in tr set class 2
for i in range(len(FMtr2)):
    for j in range(len(FMtr2[i])):
        n_tr2 += 1                              # = 515
# number of nodes in test set class 1
for i in range(len(FMtst1)):
    for j in range(len(FMtst1[i])):
        n_tst1 += 1                             # = 184
# number of nodes in test set class 2
for i in range(len(FMtst2)):
    for j in range(len(FMtst2[i])):
        n_tst2 += 1                             # = 163
# number of nodes in class 1
for i in range(50):
    for j in range(len(FM[i])):
        n_all1 += 1                             # = 810
# number of nodes in class 2
for i in range(50,100):
    for j in range(len(FM[i])):
        n_all2 += 1                             # = 847    

    
n_tr = n_tr1+ n_tr2      # = 1028
n_tst = n_tst1 + n_tst2  # = 347
n_all = n_all1 + n_all2  # = 1657

# create labels for tr set class 1
labels_tr1_n1a = np.zeros((n_tr1,1), dtype='int64')
labels_tr1_n1b = np.ones((n_tr1,1), dtype='int64')
labels_tr1_n1a_df = pd.DataFrame(labels_tr1_n1a)
labels_tr1_n1b_df = pd.DataFrame(labels_tr1_n1b)
labels_tr1_n_df = pd.concat([labels_tr1_n1a_df, labels_tr1_n1b_df], axis=1)
labels_tr1_n = labels_tr1_n_df.values # (513,2)

# create labels for tr set class 2
labels_tr2_n1a = np.ones((n_tr2,1), dtype='int64')
labels_tr2_n1b = np.zeros((n_tr2,1), dtype='int64')
labels_tr2_n1a_df = pd.DataFrame(labels_tr2_n1a)
labels_tr2_n1b_df = pd.DataFrame(labels_tr2_n1b)
labels_tr2_n_df = pd.concat([labels_tr2_n1a_df, labels_tr2_n1b_df], axis=1)
labels_tr2_n = labels_tr2_n_df.values # (515,2)

# create labels for tst set class 1
labels_tst1_n1a = np.zeros((n_tst1,1), dtype='int64')
labels_tst1_n1b = np.ones((n_tst1,1), dtype='int64')
labels_tst1_n1a_df = pd.DataFrame(labels_tst1_n1a)
labels_tst1_n1b_df = pd.DataFrame(labels_tst1_n1b)
labels_tst1_n_df = pd.concat([labels_tst1_n1a_df, labels_tst1_n1b_df], axis=1)
labels_tst1_n = labels_tst1_n_df.values # (184,2)

# create labels for tst set class 2
labels_tst2_n1a = np.ones((n_tst2,1), dtype='int64')
labels_tst2_n1b = np.zeros((n_tst2,1), dtype='int64')
labels_tst2_n1a_df = pd.DataFrame(labels_tst2_n1a)
labels_tst2_n1b_df = pd.DataFrame(labels_tst2_n1b)
labels_tst2_n_df = pd.concat([labels_tst2_n1a_df, labels_tst2_n1b_df], axis=1)
labels_tst2_n = labels_tst2_n_df.values # (163,2)

# create labels for all set class 1
labels_all1_n1a = np.zeros((n_all1,1), dtype='int64')
labels_all1_n1b = np.ones((n_all1,1), dtype='int64')
labels_all1_n1a_df = pd.DataFrame(labels_all1_n1a)
labels_all1_n1b_df = pd.DataFrame(labels_all1_n1b)
labels_all1_n_df = pd.concat([labels_all1_n1a_df, labels_all1_n1b_df], axis=1)
labels_all1_n = labels_all1_n_df.values # (810,2)

# create labels for all set class 2
labels_all2_n1a = np.ones((n_all2,1), dtype='int64')
labels_all2_n1b = np.zeros((n_all2,1), dtype='int64')
labels_all2_n1a_df = pd.DataFrame(labels_all2_n1a)
labels_all2_n1b_df = pd.DataFrame(labels_all2_n1b)
labels_all2_n_df = pd.concat([labels_all2_n1a_df, labels_all2_n1b_df], axis=1)
labels_all2_n = labels_all2_n_df.values # (847,2)

# final labels for tr set
labels_tr_n_df = pd.concat([labels_tr1_n_df, labels_tr2_n_df], axis=0)
labels_tr_n = labels_tr_n_df.values

# final labels for tst set
labels_tst_n_df = pd.concat([labels_tst1_n_df, labels_tst2_n_df], axis=0)
labels_tst_n = labels_tst_n_df.values

# final labels for complete set
labels_all_n_df = pd.concat([labels_all1_n_df, labels_all2_n_df], axis=0)
labels_all_n = labels_all_n_df.values


# save all data to prevent re-running this segment of code
np.save('labels_tr_n.npy', labels_tr_n)
np.save('labels_tst_n.npy', labels_tst_n)
np.save('labels_all_n.npy', labels_all_n)


'''
fn once n3
'''
file = open("test_idx.txt", "w")

for i in range(0,184):
    file.write(str(i))
    file.write("\n")

for i in range(1495,1658):
    file.write(str(i))
    file.write("\n")



file.close()
file = open("test_idx1000.txt", "w")

for i in range(5448,5448+1940):
    file.write(str(i))
    file.write("\n")
    
for i in range(9341+5699, 9341+5699+1903):
    file.write(str(i))
    file.write("\n")
    
file.close()

'''
fn once n4
'''
labels_tr_n = np.load('labels_tr_n.npy')   # (1028, 2)
labels_tst_n = np.load('labels_tst_n.npy') # (347, 2)
labels_all_n = np.load('labels_all_n.npy') # (1657, 2)

target1 = [0, 1]
target2 = [1, 0]

'''
With the addition of super nodes, must add 1 label per super node added
That is 50 for class 1 and 50 for class 2
Repeat with appropriate number for tr and tst sets
'''
target_all1 = [0, 1]
for i in range(49):
    target_all1 = np.vstack((target_all1, target1))

target_all2 = [1, 0]
for i in range(49):
    target_all2 = np.vstack((target_all2, target2))
    
target_tr1 = [0, 1]
for i in range(29):
    target_tr1 = np.vstack((target_tr1, target1))
    
target_tr2 = [0, 1]
for i in range(29):
    target_tr2 = np.vstack((target_tr2, target2))
    
target_tst1 = [0, 1]
for i in range(9):
    target_tst1 = np.vstack((target_tst1, target1))
    
target_tst2 = [0, 1]
for i in range(9):
    target_tst2 = np.vstack((target_tst2, target2))
    
labels_tr_n_h = np.vstack((target_tr1, labels_tr_n))
labels_tr_n_h = np.vstack((labels_tr_n_h, target_tr2)) # (1088, 2)
    
labels_tst_n_h = np.vstack((target_tst1, labels_tst_n))
labels_tst_n_h = np.vstack((labels_tst_n_h, target_tst2)) # (367, 2)

labels_all_n_h = np.vstack((target_all1, labels_all_n))
labels_all_n_h = np.vstack((labels_all_n_h, target_all2)) # (1757, 2)

# fix bug
labels_tst_n_h[357] = [1, 0]
    
np.save('labels_tr_nh.npy', labels_tr_n_h)
np.save('labels_tst_nh.npy', labels_tst_n_h)
np.save('labels_all_nh.npy', labels_all_n_h)


'''
fn once 5
dataset of length 1000
'''
n_tr1 = sum(n_rows_super[0:300])        # 5448
n_tr2 = sum(n_rows_super[500:800])      # 5699
n_tr = n_tr1 + n_tr2                    # 11147

n_tst1 = sum(n_rows_super[300:400])     # 1940
n_tst2 = sum(n_rows_super[800:900])     # 1903
n_tst = n_tst1 + n_tst2                 # 3843

n_val1 = sum(n_rows_super[400:500])     # 1953
n_val2 = sum(n_rows_super[900:1000])    # 1854
n_val = n_val1 + n_val2                 # 3807

target1 = [0, 1]
target2 = [1, 0]

# training set class 1
labels_tr1_n = target1
for i in range(n_tr1-1):
    labels_tr1_n = np.vstack((labels_tr1_n, target1))
# training set class 2
labels_tr2_n = target2
for i in range(n_tr2-1):
    labels_tr2_n = np.vstack((labels_tr2_n, target2))

# test set class 1
labels_tst1_n = target1
for i in range(n_tst1-1):
    labels_tst1_n = np.vstack((labels_tst1_n, target1))
# test set class 2
labels_tst2_n = target2
for i in range(n_tst2-1):
    labels_tst2_n = np.vstack((labels_tst2_n, target2))
    
# valid set class 1
labels_val1_n = target1
for i in range(n_val1-1):
    labels_val1_n = np.vstack((labels_val1_n, target1))
# valid set class 2
labels_val2_n = target2
for i in range(n_val2-1):
    labels_val2_n = np.vstack((labels_val2_n, target2))
    
# tr + tst + val = all
labels_all1_n = target1
for i in range(n_tr1 + n_tst1 + n_val1 - 1):
    labels_all1_n = np.vstack((labels_all1_n, target1)) # 9341
# class 2
labels_all2_n = target2
for i in range(n_tr2 + n_tst2 + n_val2 - 1):
    labels_all2_n = np.vstack((labels_all2_n, target2)) # 9456
    
# trainin set
labels_tr_n = np.vstack((labels_tr1_n, labels_tr2_n))
# test set
labels_tst_n = np.vstack((labels_tst1_n, labels_tst2_n))
# valid set
labels_val_n = np.vstack((labels_val1_n, labels_val2_n))
# all set
labels_all_n = np.vstack((labels_all1_n, labels_all2_n)) # 18797

np.save('labels1000_tr_n', labels_tr_n)
np.save('labels1000_tst_n', labels_tst_n)
np.save('labels1000_all_n', labels_all_n)