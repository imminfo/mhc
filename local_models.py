from __future__ import print_function, division

from keras.models import Model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, GRU, Bidirectional, Input, Conv1D, average, add
from keras.layers.core import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Nadam
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate
from keras.layers.advanced_activations import PReLU


def make_model_cnn1(dir_name):
    def _block(prev_layer, shape, mid_filters):
        branch = BatchNormalization()(prev_layer)
        branch = PReLU()(branch)
        branch = Conv1D(mid_filters[0], 1, kernel_initializer="he_normal")(branch)
        branch = Dropout(.3)(branch)
        
        branch = BatchNormalization()(branch)
        branch = PReLU()(branch)
        branch = Conv1D(mid_filters[1], 1, kernel_initializer="he_normal")(branch)
        branch = Dropout(.3)(branch)
        
        branch = BatchNormalization()(branch)
        branch = PReLU()(branch)
        branch = Conv1D(shape[1], 1, kernel_initializer="he_normal")(branch)
        return add([prev_layer, branch])
    
    pep_in = Input(shape=(9,20))
    pep_branch = _block(pep_in, (9,20), [128, 192])
    
    pep_branch = Flatten()(pep_branch)
    
    pep_branch = Dense(64, kernel_initializer="he_normal")(pep_branch)
    pep_branch = BatchNormalization()(pep_branch)
    pep_branch = PReLU()(pep_branch)
    pep_branch = Dropout(.5)(pep_branch)
    
    pep_branch = Dense(1)(pep_branch)
    pred = PReLU()(pep_branch)

    model = Model(pep_in, pred)
    model.compile(loss='mse', optimizer="nadam")
        
    return model


def make_model_cnn2(dir_name):
    def _block(prev_layer, shape, mid_filters):
        branch = BatchNormalization()(prev_layer)
        branch = PReLU()(branch)
        branch = Conv1D(mid_filters[0], 1, kernel_initializer="he_normal")(branch)
        branch = Dropout(.3)(branch)
        
        branch = BatchNormalization()(branch)
        branch = PReLU()(branch)
        branch = Conv1D(mid_filters[1], 1, kernel_initializer="he_normal")(branch)
        branch = Dropout(.3)(branch)
        
        branch = BatchNormalization()(branch)
        branch = PReLU()(branch)
        branch = Conv1D(shape[1], 1, kernel_initializer="he_normal")(branch)
        return add([prev_layer, branch])
    
    pep_in = Input(shape=(9,20))
    pep_branch = _block(pep_in, (9,20), [128, 192])
    
    pep_branch = Flatten()(pep_branch)
    
    pep_branch = Dense(64, kernel_initializer="he_normal")(pep_branch)
    pep_branch = BatchNormalization()(pep_branch)
    pep_branch = PReLU()(pep_branch)
    pep_branch = Dropout(.5)(pep_branch)
    
    pep_branch = Dense(64, kernel_initializer="he_normal")(pep_branch)
    pep_branch = BatchNormalization()(pep_branch)
    pep_branch = PReLU()(pep_branch)
    pep_branch = Dropout(.5)(pep_branch)
    
    pep_branch = Dense(1)(pep_branch)
    pred = PReLU()(pep_branch)

    model = Model(pep_in, pred)
    model.compile(loss='mse', optimizer="nadam")
        
    return model


def make_model_cnn3(dir_name):
    def _block(prev_layer, shape, mid_filters):
        branch = BatchNormalization()(prev_layer)
        branch = PReLU()(branch)
        branch = Conv1D(mid_filters[0], 1, kernel_initializer="he_normal")(branch)
        branch = Dropout(.3)(branch)
        
        branch = BatchNormalization()(branch)
        branch = PReLU()(branch)
        branch = Conv1D(mid_filters[1], 1, kernel_initializer="he_normal")(branch)
        branch = Dropout(.3)(branch)
        
        branch = BatchNormalization()(branch)
        branch = PReLU()(branch)
        branch = Conv1D(shape[1], 1, kernel_initializer="he_normal")(branch)
        return add([prev_layer, branch])
    
    pep_in = Input(shape=(9,20))
    pep_branch = _block(pep_in, (9,20), [64, 128])
    
    pep_branch = Flatten()(pep_branch)
    
    pep_branch = Dense(64, kernel_initializer="he_normal")(pep_branch)
    pep_branch = BatchNormalization()(pep_branch)
    pep_branch = PReLU()(pep_branch)
    pep_branch = Dropout(.5)(pep_branch)
    
    pep_branch = Dense(1)(pep_branch)
    pred = PReLU()(pep_branch)

    model = Model(pep_in, pred)
    model.compile(loss='mse', optimizer="nadam")
        
    return model


def make_model_cnn4(dir_name):
    def _block(prev_layer, shape, mid_filters):
        branch = BatchNormalization()(prev_layer)
        branch = PReLU()(branch)
        branch = Conv1D(mid_filters[0], 1, kernel_initializer="he_normal")(branch)
        branch = Dropout(.3)(branch)
        
        branch = BatchNormalization()(branch)
        branch = PReLU()(branch)
        branch = Conv1D(mid_filters[1], 1, kernel_initializer="he_normal")(branch)
        branch = Dropout(.3)(branch)
        
        branch = BatchNormalization()(branch)
        branch = PReLU()(branch)
        branch = Conv1D(shape[1], 1, kernel_initializer="he_normal")(branch)
        return add([prev_layer, branch])
    
    pep_in = Input(shape=(9,20))
    pep_branch = _block(pep_in, (9,20), [64, 128])
    
    pep_branch = Flatten()(pep_branch)
    
    pep_branch = Dense(64, kernel_initializer="he_normal")(pep_branch)
    pep_branch = BatchNormalization()(pep_branch)
    pep_branch = PReLU()(pep_branch)
    pep_branch = Dropout(.5)(pep_branch)
    
    pep_branch = Dense(64, kernel_initializer="he_normal")(pep_branch)
    pep_branch = BatchNormalization()(pep_branch)
    pep_branch = PReLU()(pep_branch)
    pep_branch = Dropout(.5)(pep_branch)
    
    pep_branch = Dense(1)(pep_branch)
    pred = PReLU()(pep_branch)

    model = Model(pep_in, pred)
    model.compile(loss='mse', optimizer="nadam")
        
    return model


def make_model_cnn5(dir_name):
    def _block(prev_layer, in_filters, out_filters):
        shortcut = BatchNormalization()(prev_layer)
        shortcut = PReLU()(shortcut)
        
        branch = Conv1D(in_filters, 1, kernel_initializer="he_normal")(shortcut)
        branch = BatchNormalization()(branch)
        branch = PReLU()(branch)
        branch = Dropout(.3)(branch)
        
        branch = Conv1D(out_filters, 1, kernel_initializer="he_normal")(branch)
        
        shortcut = Conv1D(out_filters, 1, kernel_initializer="he_normal")(shortcut)
        return add([shortcut, branch])

    char_dim=20
    max_len=9
    
    pep_in = Input(shape=(max_len, char_dim))
    pep_branch = _block(pep_in, 64, 128)
    pep_branch = _block(pep_branch, 128, 196)
    pep_branch = BatchNormalization()(pep_branch)
    pep_branch = PReLU()(pep_branch)
    
    pep_branch = Flatten()(pep_branch)

    pep_branch = Dense(64, kernel_initializer="he_normal")(pep_branch)
    pep_branch = BatchNormalization()(pep_branch)
    pep_branch = PReLU()(pep_branch)
    pep_branch = Dropout(.5)(pep_branch)

    pep_branch = Dense(1)(pep_branch)
    pred = PReLU()(pep_branch)

    model = Model(pep_in, pred)
    model.compile(loss='mse', optimizer="nadam")
    
    return model


def make_model_cnn6(dir_name):
    def _block(prev_layer, in_filters, out_filters):
        shortcut = BatchNormalization()(prev_layer)
        shortcut = PReLU()(shortcut)
        
        branch = Conv1D(in_filters, 1, kernel_initializer="he_normal")(shortcut)
        branch = BatchNormalization()(branch)
        branch = PReLU()(branch)
        branch = Dropout(.3)(branch)
        
        branch = Conv1D(out_filters, 1, kernel_initializer="he_normal")(branch)
        
        shortcut = Conv1D(out_filters, 1, kernel_initializer="he_normal")(shortcut)
        return add([shortcut, branch])

    char_dim=20
    max_len=9
    
    pep_in = Input(shape=(max_len, char_dim))
    pep_branch = _block(pep_in, 64, 128)
    pep_branch = _block(pep_branch, 128, 196)
    pep_branch = BatchNormalization()(pep_branch)
    pep_branch = PReLU()(pep_branch)
    
    pep_branch = Flatten()(pep_branch)

    pep_branch = Dense(64, kernel_initializer="he_normal")(pep_branch)
    pep_branch = BatchNormalization()(pep_branch)
    pep_branch = PReLU()(pep_branch)
    pep_branch = Dropout(.5)(pep_branch)
    
    pep_branch = Dense(64, kernel_initializer="he_normal")(pep_branch)
    pep_branch = BatchNormalization()(pep_branch)
    pep_branch = PReLU()(pep_branch)
    pep_branch = Dropout(.5)(pep_branch)

    pep_branch = Dense(1)(pep_branch)
    pred = PReLU()(pep_branch)

    model = Model(pep_in, pred)
    model.compile(loss='mse', optimizer="nadam")
    
    return model


####
####
LOCAL_MODELS = {"cnn1": make_model_cnn1, 
                "cnn2": make_model_cnn2,
                "cnn3": make_model_cnn3, 
                "cnn4": make_model_cnn4,
                "cnn5": make_model_cnn5,
                "cnn6": make_model_cnn6
               }