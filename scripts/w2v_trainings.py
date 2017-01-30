
# coding: utf-8

# In[1]:

from mhystic.embedding import *


# In[2]:

seq_for_train = get_proteins_for_embedding("../mhc/uniprot-human.fasta", fill_spaces) 


# In[28]:

params1 = {"context" : 3, "hs" : 0, "negative" : 0, "cbow_mean" : 0}

params2 = {"context" : 6, "hs" : 0, "negative" : 5, "cbow_mean" : 1}

params3 = {"context" : 9, "hs" : 0, "negative" : 5, "cbow_mean" : 0}

params4 = {"context" : 3, "hs" : 1, "negative" : 5, "cbow_mean" : 1}

params5 = {"context" : 6, "hs" : 1, "negative" : 0, "cbow_mean" : 0}

params6 = {"context" : 9, "hs" : 1, "negative" : 5, "cbow_mean" : 1}

params7 = {"context" : 3, "sg": 1, "hs" : 0, "negative" : 0}

params8 = {"context" : 6, "sg": 1, "hs" : 0, "negative" : 5}

params9 = {"context" : 9, "sg": 1, "hs" : 0, "negative" : 5}

params10 = {"context" : 3, "sg": 1, "hs" : 1, "negative" : 5}

params11 = {"context" : 6, "sg": 1,  "hs" : 1, "negative" : 0}

params12 = {"context" : 9, "sg": 1,  "hs" : 1, "negative" : 5}

params = [params1, params2, params3, params4, params5, params6,
          params7, params8, params9, params10, params11, params12]


# In[34]:

for cur_params in params:
    cur_model = train_seq2vec(seq_for_train, cur_params)
    name = str([str(i)+"_"+str(j) for i,j in cur_params.items()])
    cur_model.save("protein_model{}.pkl".format(name))


# In[ ]:



