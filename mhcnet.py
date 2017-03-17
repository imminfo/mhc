from __future__ import print_function
from keras.models import Model, load_model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, GRU, Bidirectional, Input
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate
from keras.utils.data_utils import get_file
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import numpy as np
from numpy.random import randint
import random
import sys
import re
import pandas as pd
import theano
from scipy import sparse
from sklearn.metrics import mean_squared_error, f1_score, roc_auc_score, confusion_matrix
import keras.backend as K
import matplotlib.pyplot as plt
import seaborn as sns
import pylab
import os


BIND_THR = 1 - np.log(500) / np.log(50000)


VERBOSE=2
BATCH_SIZE=32
EPOCHS=500

#theano.config.floatX="float32"
#theano.config.device="gpu1"
#theano.config.lib.cnmem="1."

def read_df(filepath):
    df = pd.read_table(filepath, sep="\t")

    df.loc[df.meas > 50000, "meas"] = 50000
    df.meas = 1 - np.log(df.meas) / np.log(50000)
    
    df.mhc = list(map(lambda x: x.replace("-", ""), df.mhc))
    df.mhc = list(map(lambda x: x.replace(":", ""), df.mhc))
    df.mhc = list(map(lambda x: x.replace("*", ""), df.mhc))

    df.loc[df.mhc == "HLAA1", "mhc"] = "HLAA0101"
    df.loc[df.mhc == "HLAA11", "mhc"] = "HLAA0101"
    df.loc[df.mhc == "HLAA2", "mhc"] = "HLAA0201"
    df.loc[df.mhc == "HLAA3", "mhc"] = "HLAA0319"
    df.loc[df.mhc == "HLAA3/11", "mhc"] = "HLAA0319"
    df.loc[df.mhc == "HLAA26", "mhc"] = "HLAA2602"
    df.loc[df.mhc == "HLAA24", "mhc"] = "HLAA2403"

    df.loc[df.mhc == "HLAB44", "mhc"] = "HLAB4402"
    df.loc[df.mhc == "HLAB51", "mhc"] = "HLAB5101"
    df.loc[df.mhc == "HLAB7", "mhc"] = "HLAB0702"
    df.loc[df.mhc == "HLAB27", "mhc"] = "HLAB2720"
    df.loc[df.mhc == "HLAB8", "mhc"] = "HLAB0801"

    df.loc[df.mhc == "HLACw1", "mhc"] = "HLAC0401"
    df.loc[df.mhc == "HLACw4", "mhc"] = "HLAC0401"

    df = df.loc[df.mhc != "HLAB60", :]
    return df
    
    
def pv_vec(seq, protvec):
    res = np.zeros((100, len(seq) - 2), dtype=float)
    for i in range(len(seq) - 2):
        res[:, i] = protvec[seq[i:i+3]]
    return res


def pv_sum(seq, protvec):
    res = np.zeros((100,), dtype=float)
    for i in range(len(seq) - 2):
        res += protvec[seq[i:i+3]]
    return res


def vectorize_mhc(seq_vec, name_vec, max_len, chars):
    res = {}
    for i, seq in enumerate(seq_vec):
        res[name_vec[i]] = np.zeros((max_len, len(chars)), dtype=np.bool)
        for row, char in enumerate(seq):
            res[name_vec[i]][row, char_indices[char]] = 1
    return res


def vectorize_xy(seq_vec, affin_vec, max_len, chars):
    X = np.zeros((len(seq_vec), max_len, len(chars)), dtype=np.bool)
    y = affin_vec
    for i, seq in enumerate(seq_vec):
        for row, char in enumerate(seq):
            X[i, row, char_indices[char]] = 1
    return X, y.reshape(len(seq_vec), 1)



#########################
# Load the ProtVec data #
#########################
"""
protvec_df = pd.read_table("data/protvec.csv", sep = "\\t", header=None)
protvec = {}
for ind, row in protvec_df.iterrows():
    row = list(row)
    row[0] = row[0][1:]
    row[-1] = row[-1][:-1]
    protvec[row[0]] = np.array(row[1:], dtype=float)
"""

    
#####################
# Prepare the chars #
#####################
chars = ["A", "L", "R", 'K', 'N', 'M', 'D', 'F', 'C', 'P', 'Q', 'S', 'E', 'T', 'G', 'W', 'H', 'Y', 'I', 'V']
print('total chars:', len(chars))
print(chars)
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))


#####################
# Load the MHC data #
#####################
print("Load MHC")
mhc_df = pd.read_csv("data/mhc_seq_imghtla.csv")
MAX_MHC_LEN = max([len(x) for x in mhc_df["pseudo"]])
X_mhc = vectorize_mhc(mhc_df["pseudo"], mhc_df["mhc"], MAX_MHC_LEN, chars)


##########################
# Load the training data #
##########################
print("Load train")
df = read_df("data/bdata.2009.tsv")
human_df = df.loc[df.species == "human", :]

MAX_PEP_LEN = max([len(x) for x in human_df["sequence"]])
X_pep_train, y_train = vectorize_xy(human_df["sequence"], human_df["meas"], MAX_PEP_LEN, chars)
X_mhc_train = np.zeros((X_pep_train.shape[0], MAX_MHC_LEN, len(chars)), dtype=np.bool)
for i, mhc in enumerate(human_df["mhc"]):
    X_mhc_train[i,:,:] = X_mhc[mhc]
print(X_pep_train.shape)
print(X_mhc_train.shape)


####################
# Load the CV data #
####################
print("Load CV")
df = read_df("data/blind.tsv")
human_df = df.loc[df.species == "human", :]

X_pep_test, y_test = vectorize_xy(human_df["sequence"], human_df["meas"], MAX_PEP_LEN, chars)
X_mhc_test = np.zeros((X_pep_test.shape[0], MAX_MHC_LEN, len(chars)), dtype=np.bool)
for i, mhc in enumerate(human_df["mhc"]):
    X_mhc_test[i,:,:] = X_mhc[mhc]
print(X_pep_test.shape)
print(X_mhc_test.shape)


###################
# Build the model #
###################
dir_name = "models/" + sys.argv[1] + "/"
if len(sys.argv) > 2:
    print("Loading model:", sys.argv[2])
    model = load_model(sys.argv[2])
else:
    if not os.path.exists(dir_name):
        print("Creating '", dir_name, "'", sep="")
        os.makedirs(dir_name)
    else:
        print(dir_name, "exists! Remove / rename it to proceed. Exiting...")
    
    mhc_in = Input(shape=(34, 20))
    mhc_branch = Bidirectional(LSTM(32, name="lstm_mhc"), "ave")(mhc_in)
    mhc_branch = Dropout(.3)(mhc_branch)
    
    pep_in = Input(shape=(15, 20))
    pep_branch = LSTM(32, name="lstm_pep"), "ave")(pep_in)
    pep_branch = Dropout(.3)(pep_branch)

    merged = concatenate([pep_branch, mhc_branch])
    merged = Dense(32)(merged)
    merged = Dropout(.3)(merged)
    merged = Dense(16)(merged)
    merged = Dropout(.3)(merged)
    merged = Dense(8)(merged)
    merged = Dropout(.3)(merged)
    pred = Dense(1, activation="relu")(merged)

    model = Model([mhc_in, pep_in], pred)
    model.compile(loss='mse', optimizer="nadam")
    
    with open(dir_name + "model.json", "w") as outf:
        outf.write(model.to_json())


print(model.summary())


###################
# Train the model #
###################
print("Training...")
for epoch in range(1, EPOCHS+1):
    history = model.fit([X_mhc_train, X_pep_train],
              y_train,
              batch_size=BATCH_SIZE,
              epochs=epoch,
              verbose=VERBOSE,
              initial_epoch=epoch-1, 
              callbacks=[ModelCheckpoint(filepath = dir_name + "model." + str(epoch % 2) + ".hdf5")])
#              validation_data=([X_mhc_test, X_pep_test], y_test))
    
    for key in history.history.keys():
        with open(dir_name + "history." + key + ".txt", "a" if epoch > 1 else "w") as hist_file:
            hist_file.writelines("\n".join(map(str, history.history[key])) + "\n")
            
    y_pred = model.predict([X_mhc_test, X_pep_test])
    
    y_true_clf = np.zeros(y_test.shape)
    y_true_clf[np.array(y_test >= BIND_THR)] = 1

    y_pred_clf = np.zeros(y_pred.shape)
    y_pred_clf[np.array(y_pred >= BIND_THR)] = 1
    
    print("F1:", f1_score(y_true_clf, y_pred_clf))
    print("AUC:", roc_auc_score(y_true_clf, y_pred_clf))
    print(confusion_matrix(y_true_clf, y_pred_clf))
    print()
    
    with open(dir_name + "history.f1.txt", "a" if epoch > 1 else "w") as hist_file:
        hist_file.writelines(str(f1_score(y_true_clf, y_pred_clf)) + "\n")
    with open(dir_name + "history.auc.txt", "a" if epoch > 1 else "w") as hist_file:
        hist_file.writelines(str(roc_auc_score(y_true_clf, y_pred_clf)) + "\n")

        
########################
# Plotting the results #
########################
"""
for iteration in range(1, 10000):
    print()
    print('-' * 50)
    print('Iteration', iteration)
#     model.fit(X, y, 
#               batch_size=128, 
#               nb_epoch=1, 
#               verbose=VERBOSE,
#               callbacks = [ModelCheckpoint(filepath = "model." + str(iteration % 2) + ".{epoch:02d}.hdf5")])
    history = model.fit_generator(generate_batch(128, iteration), 
                        samples_per_epoch=1280*3, 
                        nb_epoch=6, 
                        verbose=VERBOSE, 
                        callbacks = [ModelCheckpoint(filepath = "model." + sys.argv[1] + "." + str(iteration % 2) + ".hdf5"), 
                                     ReduceLROnPlateau(monitor="loss", factor=0.2, patience=4, min_lr=0.00001)])

    for key in history.history.keys():
        with open("history." + key + "." + sys.argv[1] + ".txt", "a" if iteration > 1 else "w") as hist_file:
            hist_file.writelines("\n".join(map(str, history.history[key])) + "\n")

    print("\nPredict big proportions:\n  real\t\tpred")
    a = y_big[:20].reshape((20,1))
    b = model.predict(X_big[:20,:,:])
    print(np.hstack([a, b]), "\n")
    
    print("Predict small proportions:\n  real\t\tpred")
    a = y_small[-20:].reshape((20,1))
    b = model.predict(X_small[-20:,:,:])
    print(np.hstack([a, b]))
"""