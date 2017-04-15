from __future__ import print_function, division
from keras.models import Model, load_model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, GRU, Bidirectional, Input, Conv1D, average
from keras.layers.core import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate
from keras.layers.advanced_activations import PReLU
from keras.utils.data_utils import get_file
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import shutil
import numpy as np
from numpy.random import randint
import random
import sys
import re
import pandas as pd
import theano
from scipy import sparse
import scipy.stats as stats 
from sklearn.metrics import mean_squared_error, f1_score, roc_auc_score, confusion_matrix
import keras.backend as K
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pylab
import os

from mhcmodel import *
from batch_generator import *


sys.setrecursionlimit(10000)


BIND_THR = 1 - np.log(500) / np.log(50000)


VERBOSE=2
BATCH_SIZE=32
EPOCHS=200
POOL_SIZE=2

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
# mhc_df = pd.read_csv("data/mhc_seq_imghtla.csv")
mhc_df = pd.read_csv("data/mhc_nature.csv")
mhc_vec = mhc_df["mhc"].unique()

mhc_map = {}
for mhc_i in range(len(mhc_df)):
    mhc_map[mhc_df["mhc"][mhc_i]] = mhc_df["pseudo"][mhc_i]
    
rev_mhc_map = {}
for mhc_i in range(len(mhc_df)):
    if mhc_df["pseudo"][mhc_i] not in rev_mhc_map:
        rev_mhc_map[mhc_df["pseudo"][mhc_i]] = []
    rev_mhc_map[mhc_df["pseudo"][mhc_i]].append(mhc_df["mhc"][mhc_i])



##########################
# Load the training data #
##########################
print("Load train")
df = read_df("data/bdata.2009.tsv")
human_df = df.loc[df.species == "human", :]
human_df = human_df.loc[human_df.peptide_length == 9, :]

MAX_PEP_LEN = max([len(x) for x in human_df["sequence"]])
X_pep_train, y_train = vectorize_xy(human_df["sequence"], human_df["meas"], MAX_PEP_LEN, chars)
print(X_pep_train.shape)

ps_arr = np.array([mhc_map[x] for x in human_df["mhc"]]).reshape((-1, 1))
ps_uniq = np.unique(ps_arr)

indices_strong = {}
indices_weak   = {}
for ps in ps_uniq:
    tmp1 = np.nonzero(np.array(y_train >= BIND_THR) & (ps_arr == ps))[0]
    tmp2 = np.nonzero(np.array(y_train < BIND_THR) & (ps_arr == ps))[0]
    if (tmp1.shape[0] >= 50) and (tmp2.shape[0] >= 50):
        indices_strong[ps] = tmp1
        indices_weak[ps]   = tmp2
        print(indices_strong[ps].shape[0], indices_weak[ps].shape[0], rev_mhc_map[ps])
    else:
        print("Skipping", tmp1.shape[0], tmp2.shape[0], rev_mhc_map[ps])
# assert(indices_strong.shape[0] + indices_weak.shape[0] == X_pep_train.shape[0])

weights_train = np.exp(stats.beta.pdf(y_train, a=3.75, b=5))


####################
# Load the CV data #
####################
print("Load CV")
df = read_df("data/blind.tsv")
human_df = df.loc[df.species == "human", :]
human_df = human_df.loc[human_df.peptide_length == 9, :]

X_pep_test, y_test = vectorize_xy(human_df["sequence"], human_df["meas"], MAX_PEP_LEN, chars)
print(X_pep_test.shape)

ps_arr = np.array([mhc_map[x] for x in human_df["mhc"]]).reshape((-1, 1))
indices_test = {}
for ps in ps_uniq:
    tmp = np.nonzero((ps_arr == ps))[0]
    if ps in indices_strong:
        indices_test[ps] = tmp
        print(indices_test[ps].shape[0], rev_mhc_map[ps])
    else:
        print("Skipping", rev_mhc_map[ps])

weights_test = np.exp(stats.beta.pdf(y_test, a=3.75, b=5))


###################
# Build the model #
###################
def make_model_cnn(dir_name):
    def _block(prev_layer, shape):
        branch = BatchNormalization()(prev_layer)
        branch = PReLU()(branch)
        # branch = Conv1D(192, 1, kernel_initializer="he_normal")(branch)
        branch = Conv1D(1, 1, kernel_initializer="he_normal")(branch)
        
        # branch = BatchNormalization()(branch)
        branch = PReLU()(branch)
        branch = Conv1D(shape[1], 1, kernel_initializer="he_normal")(branch)
        
        return add([prev_layer, branch])
    
    pep_in = Input(shape=(9,20))
    pep_branch = _block(pep_in, (9,20))
    
    pep_branch = Flatten()(pep_branch)

    pep_branch = Dense(1, kernel_initializer="he_normal")(pep_branch)
    # pep_branch = Dense(64, kernel_initializer="he_normal")(pep_branch)
    # pep_branch = BatchNormalization()(pep_branch)
    pep_branch = PReLU()(pep_branch)
    pep_branch = Dropout(.3)(pep_branch)
    
    # pep_branch = Dense(64, kernel_initializer="he_normal")(pep_branch)
    # pep_branch = BatchNormalization()(pep_branch)
    # pep_branch = PReLU()(pep_branch)
    # pep_branch = Dropout(.3)(pep_branch)
    
    pep_branch = Dense(1)(pep_branch)
    pred = PReLU()(pep_branch)

    model = Model(pep_in, pred)
    model.compile(loss='mse', optimizer="nadam")
    
    with open(dir_name + "model.json", "w") as outf:
        outf.write(model.to_json())
        
    return model

which_model, which_batch = sys.argv[1].split("_")
make_model = make_model_cnn
# if which_model == "lstm":
#     print("lstm")
#     make_model = make_model_lstm
# elif which_model == "gru":
#     print("gru")
#     make_model = make_model_gru
# elif which_model == "gru2":
#     print("gru2")
#     make_model = make_model_gru2
# elif which_model == "gruCross":
#     print("gruCross")
#     make_model = make_model_gruCross
# elif which_model == "bigru":
#     print("bigru")
#     make_model = make_model_bigru
# elif which_model == "dense":
#     print("dense")
#     make_model = make_model_dense
# elif which_model == "cnn":
#     print("cnn")
#     make_model = make_model_cnn
# elif which_model == "cnn2":
#     print("cnn2")
#     make_model = make_model_cnn2
# elif which_model == "cnn3":
#     print("cnn3")
#     make_model = make_model_cnn3
# elif which_model == "cnnrnn":
#     print("cnnrnn")
#     make_model = make_model_cnnrnn
# elif which_model == "cnnrnn2":
#     print("cnnrnn2")
#     make_model = make_model_cnnrnn2
# else:
#     print("unknown keyword model")
#     sys.exit()


dir_name = "models_local/" + sys.argv[1] + "/"
model_list = {}
if len(sys.argv) > 2:
    if sys.argv[2] == "-r":
        print("Cleaning", dir_name)
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
        os.makedirs(dir_name)
        model = make_model(dir_name)
    else:
        print("Loading model:", sys.argv[2])
        model = load_model(sys.argv[2])
else:
    if not os.path.exists(dir_name):
        print("Creating '", dir_name, "'", sep="")
        os.makedirs(dir_name)
    else:
        print(dir_name, "exists! Remove / rename it to proceed. Exiting...")
        sys.exit()
    
print("Building models...")
for ps_i, ps in enumerate(ps_uniq):
    print(ps_i, "/", len(ps_uniq) + 1, " - ", ps)
    model_list[ps] = make_model(dir_name)


# print(model.summary())


###################
# Train the model #
###################
# generate_batch = generate_batch_imba
# if which_batch == "imba":
#     print("imba")
#     generate_batch = generate_batch_imba
# elif which_batch == "bal":
#     print("bal")
#     generate_batch = generate_batch_balanced
# elif which_batch == "rand":
#     print("rand")
#     generate_batch = generate_batch_random
# elif which_batch == "wei":
#     print("wei")
#     generate_batch = generate_batch_weighted
# else:
#     print("unknown keyword batch")
#     sys.exit()

def generate_batch(X, y, batch_size, indices_strong, indices_weak):
    while True:
        to_sample_strong = batch_size // 2
        to_sample_weak   = batch_size // 2
        sampled_indices_strong = indices_strong[randint(0, indices_strong.shape[0], size=to_sample_strong)]
        sampled_indices_weak   = indices_weak[randint(0, indices_weak.shape[0], size=to_sample_weak)]
        yield np.vstack([X[sampled_indices_strong], X[sampled_indices_weak]]), \
              np.vstack([y[sampled_indices_strong], y[sampled_indices_weak]])
            

print("Training...")
EPOCHS=30
for epoch in range(1, EPOCHS+1):
    y_pred = np.zeros(y_test.shape)
    
    for ps_i, ps in enumerate(ps_uniq):
        print(rev_mhc_map[ps])
        if ps not in indices_strong.keys():
            print("skip")
            continue
        model_list[ps].fit_generator(generate_batch(X_pep_train, y_train, BATCH_SIZE, indices_strong[ps], indices_weak[ps]), 
                                           steps_per_epoch = int(X_pep_train.shape[0] / BATCH_SIZE),
                                           epochs=epoch, verbose=VERBOSE,
                                           initial_epoch=epoch-1, callbacks=[ModelCheckpoint(filepath = dir_name + "model." + str(epoch % 2) + ".hdf5")])
        
        y_pred[indices_strong[ps]] = model_list[ps].predict(X_pep_train[indices_strong[ps]])
        y_pred[indices_weak[ps]]   = model_list[ps].predict(X_pep_train[indices_weak[ps]])
    
    # for key in history.history.keys():
    #     with open(dir_name + "history." + key + ".txt", "a" if epoch > 1 else "w") as hist_file:
    #         hist_file.writelines("\n".join(map(str, history.history[key])) + "\n")
    
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