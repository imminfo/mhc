import logging
import os
import sys
import gensim
from gensim.models.doc2vec import TaggedDocument
from gensim.models import word2vec, Doc2Vec
from functools import partial
from Bio import SeqIO
import pandas as pd

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = 20

def fill_spaces(seq):
   """Put spaces between amino acids in peptide sequence
   for constructing words for embedding

   Arguments:

   seq - peptide sequence
   """
   return seq.replace("", " ")[1: -1]

def seq2vec(model, seq):
    """Performs embedding for peptide and return its vector representation

    Arguments:

    model - Word2Vec embedding model
    seq - peptide
    """
    return model[list(seq)].flatten()

def seq2protvec(model, pept):
    """Performs embedding for peptide and return its vector representation
    according to 3gram training

    model - Word2Vec embedding model
    pept - peptide
    """
    pept = list(pept)
    res = np.zeros((1, 60))
    for i in range(0, 9, 3):
        res+=seq2vec(model, pept[i:i+3])

    return res.flatten()

def iterate_uniprot_labeled_ngrams(input_data, n=3):
    """
    Create n-gram sequence generator from the primary structure in
    the FASTA format.
    :param input_data: str path to the input data
    :param n: int n-gram size
    """
    for seq_record in SeqIO.parse(input_data, "fasta"):
        ngrams = []
        sequence = str(seq_record.seq)
        for it in range(n):
            ngrams.extend(
                [sequence[i + it:i + n + it]
                 for i in range(0, len(sequence) - 1, n)
                 if len(sequence[i + it:i + n + it]) == n])
        yield TaggedDocument(words=ngrams, tags=[seq_record.id])

def train_seq2vec(data, func=fill_spaces, epochs=5, min_word_count = 10, num_workers = 15,
                  context = 9, downsampling = 1e-3, w2v_dim = 20):
    """Train the Word2Vec model on the protein primary structures data in the FASTA format.
    :param data_path: str path to the data
    :param epochs: int
    :param size: int
    :param window: int
    :param min_count: int
    :param workers: int
    :param ngram_size: int
    """
    if(isinstance(data, str)):
        sequences_for_w2v = get_proteins_for_embedding(data, func)
    else:
        sequences = data

    print("Training model...")
    w2v_model = word2vec.Word2Vec(sequences_for_w2v, workers=num_workers, size = w2v_dim,
                              min_count = min_word_count, window = context, sample = downsampling)

    print("Done.")

    return w2v_model


def train_doc2vec(data_path, epochs=15, size=100, window=3, min_count=5, workers=8, ngram_size=3):
    """
    Train the Doc2Vec model on the protein primary structures data in the FASTA format.
    :param data_path: str path to the data
    :param epochs: int
    :param size: int
    :param window: int
    :param min_count: int
    :param workers: int
    :param ngram_size: int
    """
    it = lambda: iterate_uniprot_labeled_ngrams(data_path, n=3)
    model = gensim.models.Doc2Vec(size=size, window=window, min_count=min_count, workers=workers)
    model.build_vocab(it())
    for i in range(epochs):
        model.train(it())
        logging.info("PROGRESS epoch %s from %s", i + 1, epochs)
    return model

def get_proteins_for_embedding(data_path, func):
    """Parse protein sequences from fasta file and process it in specific way with func

    Arguments:

    data_path - path where fasta file is
    func - process func for sequences(extract 3gams i.e)
    """

    list_of_sequences = []
    for seq_record in SeqIO.parse(data_path, "fasta"):
        list_of_sequences.append((str(seq_record.seq)))
    return pd.Series(list_of_sequences).apply(func)

def weight_embedding(embedding):
    return embedding/len(embedding)

def get_normed(data):

    """
    b = np.array(5*[[12, 34, 45, 23, 34, 54, 56]])

    weighted = get_normed(b)
    """
    return np.apply_along_axis(weight_embedding, 0, data)

