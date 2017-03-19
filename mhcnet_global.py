from __future__ import print_function, division
from keras.models import Model, load_model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, GRU, Bidirectional, Input, Conv1D
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
human_df = human_df.loc[human_df.peptide_length == 9, :]

MAX_PEP_LEN = max([len(x) for x in human_df["sequence"]])
X_pep_train, y_train = vectorize_xy(human_df["sequence"], human_df["meas"], MAX_PEP_LEN, chars)
X_mhc_train = np.zeros((X_pep_train.shape[0], MAX_MHC_LEN, len(chars)), dtype=np.bool)
for i, mhc in enumerate(human_df["mhc"]):
    X_mhc_train[i,:,:] = X_mhc[mhc]
print(X_pep_train.shape)
print(X_mhc_train.shape)

indices_strong = np.nonzero(np.array(y_train >= BIND_THR))[0]
indices_weak   = np.nonzero(np.array(y_train < BIND_THR))[0]
print("indices shapes:")
print(indices_strong.shape)
print(indices_weak.shape)
assert(indices_strong.shape[0] + indices_weak.shape[0] == X_pep_train.shape[0])

_, mhc_unique_indices = np.unique(mhc_df["pseudo"], return_index=True)
X_mhc_unique = np.zeros((mhc_unique_indices.shape[0], MAX_MHC_LEN, len(chars)), dtype=np.bool)
for i, j in enumerate(mhc_unique_indices):
    X_mhc_unique[i,:,:] = X_mhc[mhc_df["mhc"].loc[j]]
    
weights_train = np.exp(stats.beta.pdf(y_train, a=3.75, b=5))


####################
# Load the CV data #
####################
print("Load CV")
df = read_df("data/blind.tsv")
human_df = df.loc[df.species == "human", :]
human_df = human_df.loc[human_df.peptide_length == 9, :]

X_pep_test, y_test = vectorize_xy(human_df["sequence"], human_df["meas"], MAX_PEP_LEN, chars)
X_mhc_test = np.zeros((X_pep_test.shape[0], MAX_MHC_LEN, len(chars)), dtype=np.bool)
for i, mhc in enumerate(human_df["mhc"]):
    X_mhc_test[i,:,:] = X_mhc[mhc]
print(X_pep_test.shape)
print(X_mhc_test.shape)

weights_test = np.exp(stats.beta.pdf(y_test, a=3.75, b=5))


# X_pep_train = X_pep_train.reshape((X_pep_train.shape[0], X_pep_train.shape[1] * X_pep_train.shape[2]))
# X_mhc_train = X_mhc_train.reshape((X_mhc_train.shape[0], X_mhc_train.shape[1] * X_mhc_train.shape[2]))
# X_pep_test = X_pep_test.reshape((X_pep_test.shape[0], X_pep_test.shape[1] * X_pep_test.shape[2]))
# X_mhc_test = X_mhc_test.reshape((X_mhc_test.shape[0], X_mhc_test.shape[1] * X_mhc_test.shape[2]))

# X_train = np.hstack([X_pep_train, X_mhc_train])
# X_test = np.hstack([X_pep_test, X_mhc_test])


###################
# Build the model #
###################

def make_model_lstm(dir_name):
    mhc_in = Input(shape=(34,20))
    mhc_branch = LSTM(64)(mhc_in)
    mhc_branch = PReLU()(mhc_branch)
    
    pep_in = Input(shape=(9,20))
    pep_branch = LSTM(64)(pep_in)
    pep_branch = PReLU()(pep_branch)
    
    merged = concatenate([pep_branch, mhc_branch])
    merged = Dense(128)(merged)
    merged = Dropout(.3)(merged)
    merged = PReLU()(merged)
    
    merged = Dense(64)(merged)
    merged = Dropout(.3)(merged)
    merged = PReLU()(merged)
    
    merged = Dense(16)(merged)
    merged = Dropout(.3)(merged)
    merged = PReLU()(merged)
    
    merged = Dense(8)(merged)
    merged = Dropout(.3)(merged)
    merged = PReLU()(merged)
    
    merged = Dense(1)(merged)
    pred = PReLU()(merged)

    model = Model([mhc_in, pep_in], pred)
    model.compile(loss='mse', optimizer="nadam")
    
    with open(dir_name + "model.json", "w") as outf:
        outf.write(model.to_json())
        
    return model


def make_model_gru(dir_name):
    mhc_in = Input(shape=(34,20))
    mhc_branch = GRU(64)(mhc_in)
    mhc_branch = PReLU()(mhc_branch)
    
    pep_in = Input(shape=(9,20))
    pep_branch = GRU(64)(pep_in)
    pep_branch = PReLU()(pep_branch)
    
    merged = concatenate([pep_branch, mhc_branch])
    merged = Dense(128)(merged)
    merged = Dropout(.3)(merged)
    merged = PReLU()(merged)
    
    merged = Dense(64)(merged)
    merged = Dropout(.3)(merged)
    merged = PReLU()(merged)
    
    merged = Dense(16)(merged)
    merged = Dropout(.3)(merged)
    merged = PReLU()(merged)
    
    merged = Dense(8)(merged)
    merged = Dropout(.3)(merged)
    merged = PReLU()(merged)
    
    merged = Dense(1)(merged)
    pred = PReLU()(merged)

    model = Model([mhc_in, pep_in], pred)
    model.compile(loss='mse', optimizer="nadam")
    
    with open(dir_name + "model.json", "w") as outf:
        outf.write(model.to_json())
        
    return model


def make_model_cnn(dir_name):
    mhc_in = Input(shape=(34,20))
    mhc_branch = Conv1D(32, 5)(mhc_in)
    mhc_branch = PReLU()(mhc_branch)
    
    mhc_branch = Conv1D(32, 3)(mhc_branch)
    mhc_branch = PReLU()(mhc_branch)
    
    mhc_branch = MaxPooling1D(pool_size=POOL_SIZE)(mhc_branch)
    mhc_branch = Dropout(.2)(mhc_branch)
    
    
    pep_in = Input(shape=(9,20))
    pep_branch = Conv1D(32, 5)(pep_in)
    pep_branch = PReLU()(pep_branch)
    
    pep_branch = Conv1D(32, 3)(pep_branch)
    pep_branch = PReLU()(pep_branch)
    
    pep_branch = MaxPooling1D(pool_size=POOL_SIZE)(pep_branch)
    pep_branch = Dropout(.2)(pep_branch)
    
    
    mhc_branch = Flatten()(mhc_branch)
    pep_branch = Flatten()(pep_branch)

    
    merged = concatenate([pep_branch, mhc_branch])
    merged = Dense(128)(merged)
    merged = Dropout(.3)(merged)
    merged = PReLU()(merged)
    
    merged = Dense(64)(merged)
    merged = Dropout(.3)(merged)
    merged = PReLU()(merged)
    
    merged = Dense(16)(merged)
    merged = Dropout(.3)(merged)
    merged = PReLU()(merged)
    
    merged = Dense(8)(merged)
    merged = Dropout(.3)(merged)
    merged = PReLU()(merged)
    
    merged = Dense(1)(merged)
    pred = PReLU()(merged)

    model = Model([mhc_in, pep_in], pred)
    model.compile(loss='mse', optimizer="nadam")
    
    with open(dir_name + "model.json", "w") as outf:
        outf.write(model.to_json())
        
    return model


def make_model_dense(dir_name):
    mhc_in = Input(shape=(34,20))
    mhc_branch = Flatten()(mhc_in)
    
    mhc_branch = Dense(128)(mhc_branch)
    mhc_branch = Dropout(.3)(mhc_branch)
    mhc_branch = PReLU()(mhc_branch)
    
    mhc_branch = Dense(32)(mhc_branch)
    mhc_branch = Dropout(.3)(mhc_branch)
    mhc_branch = PReLU()(mhc_branch)
    
    
    pep_in = Input(shape=(9,20))
    pep_branch = Flatten()(pep_in)
    
    pep_branch = Dense(128)(pep_branch)
    pep_branch = Dropout(.3)(pep_branch)
    pep_branch = PReLU()(pep_branch)
    
    pep_branch = Dense(32)(pep_branch)
    pep_branch = Dropout(.3)(pep_branch)
    pep_branch = PReLU()(pep_branch)
    

    merged = concatenate([pep_branch, mhc_branch])
    merged = Dense(128)(merged)
    merged = Dropout(.3)(merged)
    merged = PReLU()(merged)
    
    merged = Dense(64)(merged)
    merged = Dropout(.3)(merged)
    merged = PReLU()(merged)
    
    merged = Dense(16)(merged)
    merged = Dropout(.3)(merged)
    merged = PReLU()(merged)
    
    merged = Dense(1)(merged)
    pred = PReLU()(merged)

    model = Model([mhc_in, pep_in], pred)
    model.compile(loss='mse', optimizer="nadam")
    
    with open(dir_name + "model.json", "w") as outf:
        outf.write(model.to_json())
        
    return model


which_model, which_batch = sys.argv[1].split("_")
make_model = make_model_lstm
if which_model == "lstm":
    make_model = make_model_lstm
elif which_model == "gru":
    make_model = make_model_gru
elif which_model == "dense":
    make_model = make_model_dense
elif which_model == "cnn":
    make_model = make_model_cnn
else:
    print("unknown keyword model")
    sys.exit()


dir_name = "models/" + sys.argv[1] + "/"
if len(sys.argv) > 2:
    if sys.argv[2] == "-r":
        print("Cleaning", dir_name)
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
    
    model = make_model(dir_name)


print(model.summary())    


###################
# Train the model #
###################
def generate_batch_imba(X_list, y, batch_size):
    while True:
        sampled_indices = randint(0, X_list[0].shape[0], size=batch_size)
        yield [X_list[0][sampled_indices], X_list[1][sampled_indices]], y[sampled_indices]

            
def generate_batch_balanced(X_list, y, batch_size):
    while True:
        to_sample_strong = batch_size / 2
        to_sample_weak   = batch_size / 2
        sampled_indices_strong = indices_strong[randint(0, indices_strong.shape[0], size=to_sample_strong)]
        sampled_indices_weak   = indices_weak[randint(0, indices_weak.shape[0], size=to_sample_weak)]
        yield [np.vstack([X_list[0][sampled_indices_strong], X_list[0][sampled_indices_weak]]), \
               np.vstack([X_list[1][sampled_indices_strong], X_list[1][sampled_indices_weak]])], \
              np.vstack([y[sampled_indices_strong], y[sampled_indices_weak]])

            
def generate_batch_random(X_list, y, batch_size):
    def rand_pep(peptide_len):
        pep = ""
        for pos in randint(0, len(chars), size=peptide_len):
            pep += chars[pos]
        return pep
    
    while True:
        to_sample_strong = int(batch_size * .4)
        to_sample_weak   = int(batch_size * .4)
        to_generate      = batch_size - to_sample_strong - to_sample_weak
        
        sampled_indices_strong = indices_strong[randint(0, indices_strong.shape[0], size=to_sample_strong)]
        sampled_indices_weak   = indices_weak[randint(0, indices_weak.shape[0], size=to_sample_weak)]
        
        X_mhc = X_mhc_unique[randint(0, X_mhc_unique.shape[0], size=to_generate)]
        X_pep, y_pep = vectorize_xy(np.array([rand_pep(X_list[1].shape[1]) for _ in range(to_generate)]), np.array([0 for _ in range(to_generate)]), X_list[1].shape[1], chars)
        
        yield [np.vstack([X_mhc, X_list[0][sampled_indices_strong], X_list[0][sampled_indices_weak]]),  \
               np.vstack([X_pep, X_list[1][sampled_indices_strong], X_list[1][sampled_indices_weak]])], \
              np.vstack([y_pep, y[sampled_indices_strong], y[sampled_indices_weak]])
            
            
def generate_batch_weighted(X_list, y, batch_size):
    while True:
        to_sample_strong = batch_size / 2
        to_sample_weak   = batch_size / 2
        sampled_indices_strong = indices_strong[randint(0, indices_strong.shape[0], size=to_sample_strong)]
        sampled_indices_weak   = indices_weak[randint(0, indices_weak.shape[0], size=to_sample_weak)]
        yield [np.vstack([X_list[0][sampled_indices_strong], X_list[0][sampled_indices_weak]]), \
               np.vstack([X_list[1][sampled_indices_strong], X_list[1][sampled_indices_weak]])], \
              np.vstack([y[sampled_indices_strong], y[sampled_indices_weak]]), \
              np.vstack([weights_train[sampled_indices_strong], weights_train[sampled_indices_weak]]).reshape((batch_size,))


generate_batch = generate_batch_imba
if which_batch == "imba":
    generate_batch = generate_batch_imba
elif which_batch == "bal":
    generate_batch = generate_batch_balanced
elif which_batch == "rand":
    generate_batch = generate_batch_random
elif which_batch == "wei":
    generate_batch = generate_batch_weighted
else:
    print("unknown keyword batch")
    sys.exit()


print("Training...")
for epoch in range(1, EPOCHS+1):
    history = model.fit_generator(generate_batch([X_mhc_train, X_pep_train], y_train, BATCH_SIZE), 
                                  steps_per_epoch = int(X_mhc_train.shape[0] / BATCH_SIZE),
                                  epochs=epoch, 
                                  verbose=VERBOSE, 
                                  initial_epoch=epoch-1, 
                                  callbacks=[ModelCheckpoint(filepath = dir_name + "model." + str(epoch % 2) + ".hdf5")])
    
    # history = model.fit([X_mhc_train, X_pep_train], y_train, 
    #                               batch_size=BATCH_SIZE,
    #                               epochs=epoch, 
    #                               verbose=VERBOSE, 
    #                               initial_epoch=epoch-1, 
    #                               callbacks=[ModelCheckpoint(filepath = dir_name + "model." + str(epoch % 2) + ".hdf5")])
    
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
        
    
    if epoch % 5 == 0:
        data_d = {}
        for file in [x for x in os.listdir(dir_name) if x.find("history") != -1]:
            title = file[8:file.rfind(".txt")]
            with open(dir_name+file) as inp:
                data_d[title] = [float(y) for y in inp.readlines()]
        print(data_d.keys())

        f, ax = plt.subplots(1,2, figsize=(16, 7))
        sns.set_style("darkgrid")
        ax[0].set_title("validation")
        ax[0].plot(data_d["f1"], label="f1")
        ax[0].plot(data_d["auc"], label="auc")
        ax[0].legend()
        ax[1].set_title("loss")
        ax[1].plot(data_d["loss"], label="loss")
        ax[1].legend()
        f.savefig(dir_name + "output.pdf")