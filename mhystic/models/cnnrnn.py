from __future__ import print_function, division
from keras.models import Model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, GRU, Bidirectional, Input, Conv1D, average, add
from keras.layers.core import Flatten
from keras.layers.pooling import MaxPooling1D, GlobalAveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate
from keras.layers.advanced_activations import PReLU


def make_model_cnnrnn(dir_name):
    def _block(prev_layer, shape):
        branch = BatchNormalization()(prev_layer)
        branch = PReLU()(branch)
        branch = Conv1D(192, 1, kernel_initializer="he_normal")(branch)
        
        branch = BatchNormalization()(branch)
        branch = PReLU()(branch)
        branch = Conv1D(shape[1], 1, kernel_initializer="he_normal")(branch)
        
        return add([prev_layer, branch])
    
    mhc_in = Input(shape=(34,20))
    mhc_branch = _block(mhc_in, (34,20)) 
    mhc_branch = GRU(32, kernel_initializer="he_normal", recurrent_initializer="he_normal", 
                       implementation=2, bias_initializer="he_normal",
                       dropout=.2, recurrent_dropout=.2, unroll=True)(mhc_branch)
    
    pep_in = Input(shape=(9,20))
    pep_branch = _block(pep_in, (9,20))
    pep_branch = GRU(32, kernel_initializer="he_normal", recurrent_initializer="he_normal", 
                       implementation=2, bias_initializer="he_normal",
                       dropout=.2, recurrent_dropout=.2, unroll=True)(pep_branch)

    merged = concatenate([pep_branch, mhc_branch])
    
    merged = Dense(64, kernel_initializer="he_normal")(merged)
    merged = BatchNormalization()(merged)
    merged = PReLU()(merged)
    merged = Dropout(.3)(merged)
    
    merged = Dense(64, kernel_initializer="he_normal")(merged)
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


def make_model_cnnrnn2(dir_name):
    def _block(prev_layer, shape):
        branch = BatchNormalization()(prev_layer)
        branch = PReLU()(branch)
        branch = Conv1D(192, 1, kernel_initializer="he_normal")(branch)
        
        branch = BatchNormalization()(branch)
        branch = PReLU()(branch)
        branch = Conv1D(shape[1], 1, kernel_initializer="he_normal")(branch)
        
        return add([prev_layer, branch])
    
    mhc_in = Input(shape=(34,20))
    mhc_branch = _block(mhc_in, (34,20)) 
    mhc_branch = _block(mhc_branch, (34,20)) 
    mhc_branch = GRU(32, kernel_initializer="he_normal", recurrent_initializer="he_normal", 
                       implementation=2, bias_initializer="he_normal",
                       dropout=.2, recurrent_dropout=.2, unroll=True)(mhc_branch)
    
    pep_in = Input(shape=(9,20))
    pep_branch = _block(pep_in, (9,20))
    pep_branch = _block(pep_branch, (9,20))
    pep_branch = GRU(32, kernel_initializer="he_normal", recurrent_initializer="he_normal", 
                       implementation=2, bias_initializer="he_normal",
                       dropout=.2, recurrent_dropout=.2, unroll=True)(pep_branch)

    merged = concatenate([pep_branch, mhc_branch])
    
    merged = Dense(64, kernel_initializer="he_normal")(merged)
    merged = BatchNormalization()(merged)
    merged = PReLU()(merged)
    merged = Dropout(.3)(merged)
    
    merged = Dense(64, kernel_initializer="he_normal")(merged)
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


def make_model_cnnrnn3(dir_name):
    def _block(prev_layer, shape):
        branch = BatchNormalization()(prev_layer)
        branch = PReLU()(branch)
        branch = Conv1D(192, 1, kernel_initializer="he_normal")(branch)
        
        branch = BatchNormalization()(branch)
        branch = PReLU()(branch)
        branch = Conv1D(shape[1], 1, kernel_initializer="he_normal")(branch)
        
        return add([prev_layer, branch])
    
    mhc_in = Input(shape=(34,20))
    mhc_branch = _block(mhc_in, (34,20)) 
    mhc_branch = _block(mhc_branch, (34,20)) 
    mhc_branch = GRU(64, kernel_initializer="he_normal", recurrent_initializer="he_normal", 
                       implementation=2, bias_initializer="he_normal",
                       dropout=.2, recurrent_dropout=.2, unroll=True)(mhc_branch)
    
    pep_in = Input(shape=(9,20))
    pep_branch = _block(pep_in, (9,20))
    pep_branch = _block(pep_branch, (9,20))
    pep_branch = GRU(64, kernel_initializer="he_normal", recurrent_initializer="he_normal", 
                       implementation=2, bias_initializer="he_normal",
                       dropout=.2, recurrent_dropout=.2, unroll=True)(pep_branch)

    merged = concatenate([pep_branch, mhc_branch])
    
    merged = Dense(64, kernel_initializer="he_normal")(merged)
    merged = BatchNormalization()(merged)
    merged = PReLU()(merged)
    merged = Dropout(.3)(merged)
    
    merged = Dense(64, kernel_initializer="he_normal")(merged)
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