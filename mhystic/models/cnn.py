from __future__ import print_function, division
from keras.models import Model
from keras.layers import Dense, Activation, Dropout
from keras.layers import Input, Conv1D, average, add
from keras.layers.core import Flatten
from keras.layers.pooling import MaxPooling1D, GlobalAveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from keras.layers.advanced_activations import PReLU


def make_model_cnn(dir_name):
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
    
    pep_in = Input(shape=(9,20))
    pep_branch = _block(pep_in, (9,20))
    
    mhc_branch = Flatten()(mhc_branch)
    pep_branch = Flatten()(pep_branch)

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


def make_model_cnn2(dir_name):
    def _block(prev_layer, shape):
        branch = BatchNormalization()(prev_layer)
        branch = PReLU()(branch)
        branch = Conv1D(192, 1, kernel_initializer="he_normal")(branch)
        
        branch = BatchNormalization()(branch)
        branch = PReLU()(branch)
        branch = Conv1D(128, 1, kernel_initializer="he_normal")(branch)
        
        branch = BatchNormalization()(branch)
        branch = PReLU()(branch)
        branch = Conv1D(shape[1], 1, kernel_initializer="he_normal")(branch)
        
        return add([prev_layer, branch])
    
    mhc_in = Input(shape=(48,20))
    mhc_branch = _block(mhc_in, (48,20))
    
    pep_in = Input(shape=(9,20))
    pep_branch = _block(pep_in, (9,20))
    
    mhc_branch = Flatten()(mhc_branch)
    pep_branch = Flatten()(pep_branch)

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


def make_model_cnn3(dir_name):
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
    
    char_dim = 20
    
    mhc_in = Input(shape=(48, char_dim))
    mhc_branch = _block(mhc_in, 64, 128)
    mhc_branch = _block(mhc_branch, 128, 196)
    mhc_branch = _block(mhc_branch, 196, 256)
    mhc_branch = BatchNormalization()(mhc_branch)
    mhc_branch = PReLU()(mhc_branch)
    # mhc_branch = GlobalAveragePooling1D()(mhc_branch)
    
    pep_in = Input(shape=(9, char_dim))
    pep_branch = _block(pep_in, 64, 128)
    pep_branch = _block(pep_branch, 128, 196)
    pep_branch = _block(pep_branch, 196, 256)
    pep_branch = BatchNormalization()(pep_branch)
    pep_branch = PReLU()(pep_branch)
    # pep_branch = GlobalAveragePooling1D()(pep_branch)
    
    mhc_branch = Flatten()(mhc_branch)
    pep_branch = Flatten()(pep_branch)

    merged = concatenate([pep_branch, mhc_branch])
    
    merged = Dense(64, kernel_initializer="he_normal")(merged)
    merged = BatchNormalization()(merged)
    merged = PReLU()(merged)
    merged = Dropout(.5)(merged)
    
    merged = Dense(64, kernel_initializer="he_normal")(merged)
    merged = BatchNormalization()(merged)
    merged = PReLU()(merged)
    merged = Dropout(.5)(merged)
    
    pred = Dense(1)(merged)
    pred = PReLU(name="pred_reg")(pred)
    
#     pred_class = Dense(1)(merged)
#     pred_class = PReLU(name="pred_clf")(pred_class)

    model = Model([mhc_in, pep_in], pred)
    model.compile(loss='mse', optimizer="nadam")
    # model.compile(loss={"pred_reg":'mse', "pred_clf": "binary_crossentropy"}, 
                  # optimizer="adam", loss_weights={"pred_reg":1, "pred_clf":.2})
    
    with open(dir_name + "model.json", "w") as outf:
        outf.write(model.to_json())
        
    return model


def make_model_cnn4(dir_name):
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
    mhc_branch = _block(mhc_branch, (34,20))
    mhc_branch = _block(mhc_branch, (34,20))
    
    pep_in = Input(shape=(9,20))
    pep_branch = _block(pep_in, (9,20))
    pep_branch = _block(pep_branch, (9,20))
    pep_branch = _block(pep_branch, (9,20))
    pep_branch = _block(pep_branch, (9,20))
    
    mhc_branch = Flatten()(mhc_branch)
    pep_branch = Flatten()(pep_branch)

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