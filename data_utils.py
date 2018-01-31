import os
import numpy as np
import pickle
#framework imports

%matplotlib inline
import matplotlib.pyplot as plt

import logging
logging.getLogger().setLevel(logging.INFO)

import numpy as np
import mxnet as mx

def cvt_data_axis(data):
    d = {}
    d["img"] = [e[0] for e in data]
    d["qst"] = [e[1] for e in data]
    d["ans"] = [e[2] for e in data]
    return d

def load_data():
    print('loading data...')
    dirs = './data'
    filename = os.path.join(dirs,'sort-of-clevr.pickle')
    with open(filename, 'rb') as f:
      train_datasets, test_datasets = pickle.load(f)
    rel_train = []
    rel_test = []
    norel_train = []
    norel_test = []
    print('processing data...')

    for img, relations, norelations in train_datasets:
        img = np.swapaxes(img,0,2)
        for qst,ans in zip(relations[0], relations[1]):
            rel_train.append((img,qst,ans))
        for qst,ans in zip(norelations[0], norelations[1]):
            norel_train.append((img,qst,ans))

    for img, relations, norelations in test_datasets:
        img = np.swapaxes(img,0,2)
        for qst,ans in zip(relations[0], relations[1]):
            rel_test.append((img,qst,ans))
        for qst,ans in zip(norelations[0], norelations[1]):
            norel_test.append((img,qst,ans))
    
    return (cvt_data_axis(rel_train), cvt_data_axis(rel_test), cvt_data_axis(norel_train), cvt_data_axis(norel_test))