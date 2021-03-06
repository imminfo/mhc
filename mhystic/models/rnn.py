from __future__ import print_function, division
from keras.models import Model, load_model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, GRU, Bidirectional, Input, Conv1D, average, add
from keras.layers.core import Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Nadam


def make_model_lstm(dir_name):
    mhc_in = Input(shape=(34,20))
    mhc_branch = LSTM(64, kernel_initializer="he_uniform")(mhc_in)
    mhc_branch = PReLU()(mhc_branch)
    
    pep_in = Input(shape=(9,20))
    pep_branch = LSTM(64, kernel_initializer="he_uniform")(pep_in)
    pep_branch = PReLU()(pep_branch)
    
    merged = concatenate([pep_branch, mhc_branch])
    merged = Dense(128, kernel_initializer="he_uniform")(merged)
    merged = Dropout(.3)(merged)
    merged = PReLU()(merged)
    
    merged = Dense(64, kernel_initializer="he_uniform")(merged)
    merged = Dropout(.3)(merged)
    merged = PReLU()(merged)
    
    merged = Dense(16, kernel_initializer="he_uniform")(merged)
    merged = Dropout(.3)(merged)
    merged = PReLU()(merged)
    
    merged = Dense(8, kernel_initializer="he_uniform")(merged)
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
    mhc_branch = GRU(64, kernel_initializer="he_normal", recurrent_initializer="he_normal", 
                       implementation=2, bias_initializer="he_normal",
                       dropout=.2, recurrent_dropout=.2, unroll=True)(mhc_in)
    mhc_branch = BatchNormalization()(mhc_branch)
    mhc_branch = PReLU()(mhc_branch)
    
    pep_in = Input(shape=(9,20))
    pep_branch = GRU(64, kernel_initializer="he_normal", recurrent_initializer="he_normal", 
                       implementation=2, bias_initializer="he_normal",
                       dropout=.2, recurrent_dropout=.2, unroll=True)(pep_in)
    pep_branch = BatchNormalization()(pep_branch)
    pep_branch = PReLU()(pep_branch)
    
    merged = concatenate([pep_branch, mhc_branch])
    merged = Dense(64, kernel_initializer="he_uniform")(merged)
    merged = BatchNormalization()(merged)
    merged = PReLU()(merged)
    merged = Dropout(.3)(merged)
    
    merged = Dense(64, kernel_initializer="he_uniform")(merged)
    merged = BatchNormalization()(merged)
    merged = PReLU()(merged)
    merged = Dropout(.3)(merged)
    
    merged = Dense(1)(merged)
    pred = PReLU()(merged)

    model = Model([mhc_in, pep_in], pred)
    model.compile(loss='mse', optimizer="nadam")
    
    with open(dir_name + "model.json", "w") as outf:
        outf.write(model.to_json())
        
    return model


def make_model_gru2(dir_name):
    mhc_in = Input(shape=(34,20))
    mhc_branch = GRU(32, kernel_initializer="he_normal", recurrent_initializer="he_normal", 
                       implementation=2, bias_initializer="he_normal",
                       dropout=.2, recurrent_dropout=.2, unroll=True, return_sequences=True)(mhc_in)
    mhc_branch = GRU(32, kernel_initializer="he_normal", recurrent_initializer="he_normal", 
                   implementation=2, bias_initializer="he_normal",
                   dropout=.2, recurrent_dropout=.2, unroll=True)(mhc_branch)
    mhc_branch = BatchNormalization()(mhc_branch)
    mhc_branch = PReLU()(mhc_branch)
    
    pep_in = Input(shape=(9,20))
    pep_branch = GRU(32, kernel_initializer="he_normal", recurrent_initializer="he_normal", 
                       implementation=2, bias_initializer="he_normal",
                       dropout=.2, recurrent_dropout=.2, unroll=True, return_sequences=True)(pep_in)    
    pep_branch = GRU(32, kernel_initializer="he_normal", recurrent_initializer="he_normal", 
                       implementation=2, bias_initializer="he_normal",
                       dropout=.2, recurrent_dropout=.2, unroll=True)(pep_branch)
    pep_branch = BatchNormalization()(pep_branch)
    pep_branch = PReLU()(pep_branch)
    
    merged = concatenate([pep_branch, mhc_branch])
    merged = Dense(64, kernel_initializer="he_uniform")(merged)
    merged = BatchNormalization()(merged)
    merged = PReLU()(merged)
    merged = Dropout(.3)(merged)
    
    merged = Dense(64, kernel_initializer="he_uniform")(merged)
    merged = BatchNormalization()(merged)
    merged = PReLU()(merged)
    merged = Dropout(.3)(merged)
    
    merged = Dense(1)(merged)
    pred = PReLU()(merged)

    model = Model([mhc_in, pep_in], pred)
    model.compile(loss='mse', optimizer="nadam")
    
    with open(dir_name + "model.json", "w") as outf:
        outf.write(model.to_json())
        
    return model


def make_model_bigru(dir_name):
    gru_node = lambda x: GRU(64, kernel_initializer="he_normal", recurrent_initializer="he_normal", 
                                 implementation=2, bias_initializer="he_normal",
                                 dropout=.2, recurrent_dropout=.2, unroll=True, go_backwards = x)
    
    # MHC BiGRU
    mhc_in = Input(shape=(34,20))
    mhc_branch_forw = gru_node(False)(mhc_in)
    mhc_branch_forw = BatchNormalization()(mhc_branch_forw)
    mhc_branch_forw = PReLU()(mhc_branch_forw)
    
    mhc_branch_back = gru_node(True)(mhc_in)
    mhc_branch_back = BatchNormalization()(mhc_branch_back)
    mhc_branch_back = PReLU()(mhc_branch_back)
    
    mhc_branch = average([mhc_branch_forw, mhc_branch_back])
    
    # Peptide BiGRU
    pep_in = Input(shape=(9,20))
    pep_branch_forw = gru_node(False)(pep_in)
    pep_branch_forw = BatchNormalization()(pep_branch_forw)
    pep_branch_forw = PReLU()(pep_branch_forw)
    
    pep_branch_back = gru_node(True)(pep_in)
    pep_branch_back = BatchNormalization()(pep_branch_back)
    pep_branch_back = PReLU()(pep_branch_back)
    
    pep_branch = average([pep_branch_forw, pep_branch_back])
    
    # Merge branches
    merged = concatenate([pep_branch, mhc_branch])
    merged = Dense(64, kernel_initializer="he_uniform")(merged)
    merged = BatchNormalization()(merged)
    merged = PReLU()(merged)
    merged = Dropout(.3)(merged)
    
    merged = Dense(64, kernel_initializer="he_uniform")(merged)
    merged = BatchNormalization()(merged)
    merged = PReLU()(merged)
    merged = Dropout(.3)(merged)
    
    pred = Dense(1)(merged) #, activation = "sigmoid"
    pred = PReLU()(pred)

    model = Model([mhc_in, pep_in], pred)
    model.compile(loss='kullback_leibler_divergence', optimizer="nadam")
    
    with open(dir_name + "model.json", "w") as outf:
        outf.write(model.to_json())
        
    return model