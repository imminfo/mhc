from __future__ import print_function, division
from keras.models import Model
from keras.layers import Dense, Activation, Dropout
from keras.layers import Input, average, add
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Nadam


def make_model_dense(dir_name):
    mhc_in = Input(shape=(34,20))
    mhc_branch = Flatten()(mhc_in)
    
    mhc_branch = Dense(64, kernel_initializer="he_normal")(mhc_branch)
    mhc_branch = BatchNormalization()(mhc_branch)
    mhc_branch = PReLU()(mhc_branch)
    mhc_branch = Dropout(.3)(mhc_branch)
    
    # mhc_branch = Dense(32, kernel_initializer="he_normal")(mhc_branch)
    # mhc_branch = BatchNormalization()(mhc_branch)
    # mhc_branch = PReLU()(mhc_branch)
    # mhc_branch = Dropout(.3)(mhc_branch)

    
    pep_in = Input(shape=(9,20))
    pep_branch = Flatten()(pep_in)
    
    pep_branch = Dense(64, kernel_initializer="he_normal")(pep_branch)
    pep_branch = BatchNormalization()(pep_branch)
    pep_branch = PReLU()(pep_branch)
    pep_branch = Dropout(.3)(pep_branch)
    
    # pep_branch = Dense(32, kernel_initializer="he_normal")(pep_branch)
    # pep_branch = BatchNormalization()(pep_branch)
    # pep_branch = PReLU()(pep_branch)
    # pep_branch = Dropout(.3)(pep_branch)
    

    merged = concatenate([pep_branch, mhc_branch])
    merged = Dense(128, kernel_initializer="he_normal")(merged)
    merged = BatchNormalization()(merged)
    merged = PReLU()(merged)
    merged = Dropout(.3)(merged)
    
    # merged = Dense(128, kernel_initializer="he_normal")(merged)
    # merged = BatchNormalization()(merged)
    # merged = PReLU()(merged)
    # merged = Dropout(.3)(merged)
    
    merged = Dense(1)(merged)
    pred = PReLU()(merged)

    model = Model([mhc_in, pep_in], pred)
    model.compile(loss='mse', optimizer="nadam")
    
    with open(dir_name + "model.json", "w") as outf:
        outf.write(model.to_json())
        
    return model