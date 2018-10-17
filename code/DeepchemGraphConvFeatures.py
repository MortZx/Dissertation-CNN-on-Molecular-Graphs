# -*- coding: utf-8 -*-
"""
Ran on Windows virtual machine 
as DeepChem only compatible with LINUX.

Run MoleculeNet's graph conv 
Extract all feature vectors here
"""

#from __future__ import division
#from __future__ import print_function
#from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
import deepchem as dc
from deepchem.models.tensorgraph.models.graph_models import GraphConvTensorGraph
import pandas as pd

from IPython.display import Image, display

# Load Tox21 dataset
# MoleculeNet call returns tr-tst-val sets
# call also returns 'transformers', list of data transforms applied to pre-processed data
n_features = 1024
tox21_tasks, tox21_datasets, transformers = dc.molnet.load_tox21(featurizer='GraphConv')
''' output:
Loading dataset from disk
Loading dataset from disk
Loading dataset from disk
'''
# simply divide tox21_datasets into train, valid and test
train_dataset, valid_dataset, test_dataset = tox21_datasets

tr_w = getattr(train_dataset, "w")
tr_ids = getattr(train_dataset, "ids")
tr_y = getattr(train_dataset, "y")
tr_X = getattr(train_dataset, "X")

tr_X0 = tr_X[10]
ids_tr = tr_ids[10]
smile1 = 'C[C@]1(O)CC[C@H]2[C@@H]3CCC4=CC(=O)CC[C@]4(C)[C@@]3(F)[C@@H](O)C[C@@]21C'
smile2 = 'CCOc1ccc2nc(S(N)(=O)=O)sc2c1'


#X_obj = getattr(train_dataset, "X")[index]

###############################################################################
'''
Visualise first x molecules as graphs from strings using RDKit
Each molecule graph is saved as a png file
'''
from rdkit import Chem
from rdkit.Chem import Draw
from itertools import islice
from IPython.display import Image, display, HTML

def display_images(filenames):
    """Helper to pretty-print images."""
    for filename in filenames:
        display(Image(filename))
        
def mols_to_pngs(mols, basename="test"):
    """Helper to write RDKit mols to png files."""
    filenames = []
    for i, mol in enumerate(mols):
        filename = "MUV_%s%d.png" % (basename, i)
        Draw.MolToFile(mol, filename)
        filenames.append(filename)
    return filenames

molecules=[]
molecules.append(Chem.MolFromSmiles(ids_tr)) # ids_tr instead of smile
#display_images(mols_to_pngs(molecules))

#df = pd.DataFrame(train_dataset)
#pprint(vars(train_dataset))

#import os
#dir = 'C:/Users/MortZ/Desktop/data'
#dc.utils.save.save_dataset_to_disk(dir, train_dataset, valid_dataset, test_dataset, transformers

# GraphConvTensorGraph wraps a standard GC architecture under the hood
# instantiate object this class
model = GraphConvTensorGraph(len(tox21_tasks), batch_size=50, mode='classification', tensorboard=True)
# Set nb_epoch=10 for better results.
model.fit(train_dataset, nb_epoch=20)
''' output:
/home/mortz/anaconda3/envs/MoleculeNet/lib/python3.5/site-packages/tensorflow/python/ops/gradients_impl.py:97: 
    UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. 
    This may consume a large amount of memory.
  "Converting sparse IndexedSlices to a dense Tensor of unknown shape. "
Starting epoch 0
Starting epoch 1
Starting epoch 2
Starting epoch 3
Starting epoch 4
Starting epoch 5
Starting epoch 6
Starting epoch 7
Starting epoch 8
Starting epoch 9
Out[6]: 388.09922670678617
'''

###############################################################################
'''
Viewing Tensorboard output
'''
# When param tensorboard=True when calling model=GraphConvTensorGraph(), 
# all data files logged to model.model_dir

# model.model_dir = '/mnt/c/Users/MortZ/Desktop/cnnChemoinfoProject/data/sudo'

# launch tensorboard webserver by imputting in terminal:
#tensorboard --logdir=model.model_dir    # launches on port 6006
# open http://localhost6006
#display(Image(filename='assets/tensorboard_landing.png'))

###############################################################################

# Use ROC-AUC metric to measure model performance
# (tradeoff precision Vs recall)
metric = dc.metrics.Metric( dc.metrics.accuracy_score, np.mean, mode="classification")

print("Evaluating model")
train_scores = model.evaluate(train_dataset, [metric], transformers)
''' output
train_scores = model.evaluate(train_dataset, [metric], transformers)
computed_metrics: [0.91303328898580394, 0.9613695207129096, 0.91368501236645505, 
                   0.92415904273922433, 0.80878436783137742, 0.92090058205232073, 
                   0.95985605332514923, 0.83400977463114989, 0.93741430788611035, 
                   0.88537550092812656, 0.91697844508627457, 0.9149230855288919]

computed_metrics: [0.91024081155978998, 0.95949416947655752, 0.91102686198016269, 
                   0.9211747696576924, 0.80632020863577758, 0.91459110646795871, 
                   0.95547735000282663, 0.847896947938373, 0.93525939046372919, 
                   0.89491907083253819, 0.91385082807930584, 0.91380432958550095]
ACC
-----
computed_metrics: [0.74091300602928511, 0.92047565960609434, 0.82376578645235365, 
                   0.81272923408845743, 0.59610310533793387, 0.8728553368249955, 
                   0.86465433300876338, 0.74325481798715198, 0.87717429889953846, 
                   0.8540540540540541, 0.77672209026128269, 0.87176165803108807]

computed_metrics: [0.76881998277347119, 0.839836492010405, 0.68905472636815923, 
                   0.86170442286947146, 0.66287801907854682, 0.82120281741014989, 
                   0.86407010710808174, 0.69571734475374736, 0.80475683351082716, 
                   0.65501930501930505, 0.78967825523644997, 0.67264988897113254]
'''

print("Training ROC-AUC Score: %f" % train_scores["mean-accuracy_score"])
''' Training ROC-AUC Score: 0.907005 

Training ROC-AUC Score: 0.760449
Training ROC-AUC Score: 0.812872
Training ROC-AUC Score: 0.860603
Training ROC-AUC Score: 0.812211
Training ROC-AUC Score: 0.698896'''

valid_scores = model.evaluate(valid_dataset, [metric], transformers)
''' output
computed_metrics: [0.81224954237371993, 0.82895171957671954, 0.87816866153916173, 
                   0.82219115404168785, 0.68327272727272725, 0.7694909920541263, 
                   0.74169262720664597, 0.84133023321355593, 0.83329408994584409, 
                   0.77308072635742531, 0.8650746125655705, 0.88087855297157636]

computed_metrics: [0.79646761984861225, 0.8252314814814814, 0.85782551312975019, 
                   0.79658108795119476, 0.72629545454545452, 0.79238454881598619, 
                   0.77301834544825199, 0.83538833697362547, 0.85910054155874738, 
                   0.77123786407766992, 0.86893099235909832, 0.836046511627907]

ACC
----
computed_metrics: [0.76446280991735538, 0.80724637681159417, 0.66370370370370368, 
                   0.85254237288135593, 0.63174603174603172, 0.81895332390381892, 
                   0.83787878787878789, 0.68120805369127513, 0.78212290502793291, 
                   0.62079510703363916, 0.76109215017064846, 0.67400881057268724]

computed_metrics: [0.74793388429752061, 0.92173913043478262, 0.81333333333333335, 
                   0.8203389830508474, 0.55238095238095242, 0.85148514851485146, 
                   0.8545454545454545, 0.73322147651006708, 0.87011173184357538, 
                   0.84556574923547401, 0.76279863481228671, 0.85315712187958881]
'''

print("Validation ROC-AUC Score: %f" % valid_scores["mean-accuracy_score"])
''' output
Validation ROC-AUC Score: 0.810806

Validation ROC-AUC Score: 0.811542
ACC ----
Validation ROC-AUC Score: 0.741313
Validation ROC-AUC Score: 0.802218
Validation ROC-AUC Score: 0.850613
Validation ROC-AUC Score: 0.797486
Validation ROC-AUC Score: 0.685671
'''
