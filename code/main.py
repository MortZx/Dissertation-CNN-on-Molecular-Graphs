# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 02:06:34 2018

@author: MortZ

Tox21 data challenge for 1000 instances
"""

from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from gcn.utils import *
from gcn.inits import *
from gcn.metrics import *
#from gcn.models import GCN, MLP
from data_loader import *

from helpers import *
import pprint
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.linalg as sl

from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

###############################################################################
# load data
adj, features, idx_super, y_train, y_test, y_val, train_mask, test_mask, test_mask, val_mask = load_data()

# create index list of with super nodes
idx_super2 = [ [] for x in range(len(idx_super))]
for i in range(len(idx_super)):
    idx_super2[i] = idx_super[i]

idx_super2 = tf.constant(idx_super2)

                
       
###############################################################################

'''
MODEL
'''

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)


################################
# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}

def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]
    
def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)

def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)

def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all), correct_prediction

def super_acc(preds, labels):
    pass
    

def get_super(all_nodes):
    #all_nodes = all_nodes.eval()
    super_n = [ [] for x in range(1000)]
    a = 0
    for i in range(18797):
        for j in range(len(idx_super)):
            if i == idx_super[j]:
                super_n[a] = all_nodes[i]
                a += 1
    super_n[-1] = all_nodes[-1]
    #super_n = tf.convert_to_tensor(super_n, dtype=tf.int64)#, dtype=tf.float32)
    return super_n


    

class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).
    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off
    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs) ## CONVOLUTION
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])
            
class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs # bool
        self.featureless = featureless # bool
        self.bias = bias # bool

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)): # self.vars['weights_0'] has shape=(1433,16)
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],# gc2 weights_0 has
                                                        name='weights_' + str(i))# shape=(16,7)
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None
        self.final = []

        self.loss = 0
        self.accuracy = 0
        self.correct_preds = 0
        self.optimizer = None
        self.opt_op = None


    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs) # before this line self.activations = []
        for layer in self.layers: # number of gc layers so = 2
            hidden = layer(self.activations[-1]) # gc1 w/ relu (?,16) // gc2 SparseTensorDenseMatMul (?,7)
            self.activations.append(hidden)  ## CALL CONVOLUTION
        self.outputs = self.activations[-1]
        #self.outputs = tf.gather(params=self.outputs, indices=idx_super2)

        # Store model variables for easy access
        gl_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name) # scope = 'gcn'
        l_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=self.name)
        tr_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope=self.name)
        w_variables = tf.get_collection(tf.GraphKeys.WEIGHTS, scope=self.name)
        
        self.gl_vars = {var.name: var for var in gl_variables}
        self.l_vars = {var.name: var for var in l_variables}
        self.tr_vars = {var.name: var for var in tr_variables}
        self.vars = {var.name: var for var in variables}
        self.w_vars = {var.name: var for var in w_variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)



class GCN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values(): # = len(1)
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var) # 0.0005 * 

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        #print('\n\noutputs!', tf.Tensor.eval(self.outputs), '\n\n')
        self.accuracy, self.correct_preds = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim,   #1433
                                            output_dim=FLAGS.hidden1,   #16
                                            #output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=False,
                                            sparse_inputs=True,
                                            logging=self.logging,))
        
        def predict(self):
            print("self.outputs = ", self.outputs)
            final = tf.nn.softmax(self.outputs)
        
            global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name) # scope = 'gcn'
            local_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=self.name)
            train_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
            all_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope=self.name)
            weight_variables = tf.get_collection(tf.GraphKeys.WEIGHTS, scope=self.name)
        
            self.global_vars = {var.name: var for var in global_variables}
            self.local_vars = {var.name: var for var in local_variables}
            self.train_vars = {var.name: var for var in train_variables}
            self.all_vars = {var.name: var for var in all_variables}
            self.weight_vars = {var.name: var for var in weight_variables}
        
            return final
        
        #self.layers.append(predict(self))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=False,
                                            logging=self.logging))


class SelfAttention(Layer):
    def __init__(self, attention_dim, bias_dim, hidden_units, **kwargs):
        super().__init__(**kwargs)
        self.hidden_units = hidden_units
        self.A = None
        self.vars['Ws'] = tf.Variable(tf.random_uniform([attention_dim, self.hidden_units]))
        self.vars['W2'] = tf.Variable(tf.random_uniform([bias_dim, attention_dim]))

    def _call(self, inputs):
        aux = tf.tanh(tf.matmul(self.vars['Ws'], inputs, transpose_b=True))
        self.A = tf.nn.softmax(tf.matmul(self.vars['W2'], aux))
        tf.summary.histogram('self_attention', self.A)
        out = tf.matmul(self.A, inputs)
        out = tf.reshape(out, [out.get_shape().as_list()[0] * out.get_shape().as_list()[1]])
        return out



_AM, _FM, labels = load_tox21_SR_MMP()
labels1000 = [ [] for x in range(1000)]
for i in range(500):
      labels1000[i] = labels[i]   

for i in range(500):
    labels1000[i+500] = labels[i+4892]

labels = np.array(labels1000)


###############################################################################

# del_all_flags(tf.flags.FLAGS)
# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 5, 'Tolerance for early stopping (# of epochs).')


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.float_power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


# Some preprocessing
features = preprocess_features(features)
support = [preprocess_adj(adj)]
num_supports = 1
      

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

model = GCN(placeholders, input_dim=features[2][1], logging=True)

# Initialize session
sess = tf.Session()
#with sess.as_default():
    # Create model (model_func = GCN)
    

# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()    
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy, model.correct_preds], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], outs_val[2], (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())
#tf.global_variables_initializer()

cost_val = []

# Train model
loss = [ [] for x in range(200)]
val_loss = [ [] for x in range(200)]
acc_tr = [ [] for x in range(200)]
acc_val = [ [] for x in range(200)]
acc_tst = [ [] for x in range(200)]
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    _, m_loss, m_acc, m_outputs = sess.run([model.opt_op, model.loss, model.accuracy, model.outputs], feed_dict=feed_dict)
    
    # Validation
    cost, acc, cor_preds, duration = evaluate(features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)
    
    test_cost, test_acc, test_correct, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
    
    loss[epoch] = m_loss
    val_loss[epoch] = cost
    acc_tr[epoch] = m_acc
    acc_val[epoch] = acc
    acc_tst[epoch] = test_acc

    # Print results
    '''
    print("Epoch:", '%04d' % (epoch + 1), "train_acc=", "{:.5f}".format(m_acc),
          "val_acc=", "{:.5f}".format(acc), "tst_acc=", "{:.5f}".format(test_acc))
    '''
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(m_loss),
          "train_acc=", "{:.5f}".format(m_acc), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "tst_acc=", "{:.5f}".format(test_acc))#, "time=", "{:.5f}".format(time.time() - t))

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        #print("Early stopping...")
        break

#print("Optimization Finished!")

# Testing
#test_cost, test_acc, test_correct, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
#print("Test set results:", "cost=", "{:.5f}".format(test_cost),
 #     "accuracy=", "{:.5f}".format(test_acc))#, "time=", "{:.5f}".format(test_duration))


plt.title('Model Loss')
plt.plot(loss[0:epoch+1], label='training loss')
plt.plot(val_loss[0:epoch+1], label='validation loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(loc='upper right')
plt.grid()
plt.show()

plt.title('Model Accuracy')
#plt.plot(acc_val[0:epoch+1], label='validation accuracy')
plt.plot(acc_tr[0:epoch+1], label='train accuracy')
plt.plot(acc_tst[0:epoch+1], label='test accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.grid()
plt.legend(loc='lower right')
plt.show()


'''
IMPORTANT
this function must be run each time the model is run!
'''
del_all_flags(tf.flags.FLAGS)




graph = [ [] for x in range(1000)]
a = 0
for i in range(len(m_outputs)):
    for j in range(len(idx_super)):
        if i == idx_super[j]:
            graph[a] = m_outputs[i]
            a += 1
graph[-1] = m_outputs[-1]
graph = np.array(graph)

import sklearn as sk
from sklearn import preprocessing
from sklearn import metrics

m_norm = sk.preprocessing.normalize(graph)



# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i], _ = roc_curve(labels[:, i], m_norm[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
 

# plot ROC both classes
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(2)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(2):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= 2
lw=1


# Plot all ROC curves
plt.figure()
plt.plot(fpr[0], tpr[0], color='blue', lw=lw,
             label='class 0: not toxic (area = {1:0.2f})'
             ''.format(0, roc_auc[0]))
plt.plot(fpr[1], tpr[1], color='red', lw=lw,
             label='class 1: toxic (area = {1:0.2f})'
             ''.format(1, roc_auc[1]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Both Classes')
plt.legend(loc="lower right")
plt.show()