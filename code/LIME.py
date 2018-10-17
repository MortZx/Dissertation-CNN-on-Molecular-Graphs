'''
Ran on Windows virtual machine 
as DeepChem only compatible with LINUX.

Run MoleculeNet's ECFP

Implement LIME to obtain eplanations
'''

# Imaging imports to get pictures in the notebook
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from IPython.display import SVG
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D


import numpy as np
import deepchem as dc
from deepchem.molnet import load_tox21

# Only for debug!
np.random.seed(123)

# Load Tox21 dataset
n_features = 1024
tox21_tasks, tox21_datasets, transformers = load_tox21()
train_dataset, valid_dataset, test_dataset = tox21_datasets

tr_w = getattr(train_dataset, "w")
tr_ids = getattr(train_dataset, "ids")
tr_y = getattr(train_dataset, "y")
tr_X = getattr(train_dataset, "X")

tst_y = getattr(test_dataset, "y")

# Fit models
metric = dc.metrics.Metric(
    dc.metrics.roc_auc_score, np.mean, mode="classification")

nb_epoch = 10
model = dc.models.tensorgraph.fcnet.MultiTaskClassifier(
    len(tox21_tasks),
    train_dataset.get_data_shape()[0])

# Fit trained model
model.fit(train_dataset, nb_epoch=nb_epoch)
model.save()

print("Evaluating model")
train_scores = model.evaluate(train_dataset, [metric], transformers)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)
test_scores = model.evaluate(test_dataset, [metric], transformers)

print("Train scores")
print(train_scores)

print(test_scores)

print("Validation scores")
print(valid_scores)

from lime import lime_tabular
feature_names = ["fp_%s"  % x for x in range(1024)]
explainer = lime_tabular.LimeTabularExplainer(train_dataset.X, 
                                              feature_names=feature_names, 
                                              categorical_features=feature_names,
                                              class_names=['not toxic', 'toxic'], 
                                              discretize_continuous=True)

# need a function which takes a 2d numpy array (samples, features) and returns predictions (samples,)
def eval_model(my_model, transformers):
    def eval_closure(x):
        ds = dc.data.NumpyDataset(x, None, None, None)
        # The 0th task is NR-AR
        predictions = model.predict_proba(ds)[:,0]
        return predictions
    return eval_closure
model_fn = eval_model(model, transformers)

# We want to investigate a toxic compound
active_id = np.where(test_dataset.y[:,0]==1)[0][20]
print("active ID: ", active_id)
print("SMILES: ", test_dataset.ids[active_id])
Chem.MolFromSmiles(test_dataset.ids[active_id])

# this returns an Lime Explainer class
# The explainer contains details for why the model behaved the way it did
exp = explainer.explain_instance(test_dataset.X[active_id], model_fn, num_features=5, top_labels=1)

# If in an ipython notebook it can show it 
exp.show_in_notebook(show_table=True, show_all=False)



from rdkit import Chem

def fp_mol(mol, fp_length=1024):
    """
    returns: dict of <int:list of string>
        dictionary mapping fingerprint index
        to list of smile string that activated that fingerprint
    """
    d = {}
    feat = dc.feat.CircularFingerprint(sparse=True, smiles=True, size=1024)
    retval = feat._featurize(mol)
    for k, v in retval.items():
        index = k % 1024
        if index not in d:
            d[index] = set()
        d[index].add(v['smiles'])
    return d
# What fragments activated what fingerprints in the molecule
my_fp = fp_mol(Chem.MolFromSmiles(test_dataset.ids[active_id]))

# can calculate which fragments activate all fingerprint
# indexes throughout entire training set
all_train_fps = {}
X = train_dataset.X
ids = train_dataset.ids
for i in range(len(X)):
    d = fp_mol(Chem.MolFromSmiles(ids[i]))
    for k, v in d.items():
        if k not in all_train_fps:
            all_train_fps[k] = set()
        all_train_fps[k].update(v)
        
#can visualize which fingerprints model declared toxic for the
# active molecule investigating
Chem.MolFromSmiles(list(my_fp[118])[0])
Chem.MolFromSmiles(list(my_fp[519])[0])
Chem.MolFromSmiles(list(my_fp[301])[0])

# can also see what fragments are missing by investigating the training set 
# According to  explanation having one of these fragments would make molecule more
# likely to be toxic
Chem.MolFromSmiles(list(all_train_fps[381])[0])
Chem.MolFromSmiles(list(all_train_fps[381])[1])
Chem.MolFromSmiles(list(all_train_fps[381])[4])
Chem.MolFromSmiles(list(all_train_fps[875])[0])
Chem.MolFromSmiles(list(all_train_fps[875])[2])
Chem.MolFromSmiles(list(all_train_fps[875])[4])
# Using LIME on fragment based models can give intuition over which fragments 
# are contributing to response variable in a linear fashion.