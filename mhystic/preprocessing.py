import math
import pandas as pd
import numpy as np
from pandas import get_dummies
from sklearn.cluster import KMeans
import re

def log_meas(affinity):
    """Logarithming according to article. y_i = 1-log(affinity)/log(50,000)

    Arguments:

    affinity - IC50 affinity in nm
    """
    new_meas = 1 - math.log(affinity, 50000.0)
    new_meas = np.clip(new_meas, 0.0, 1.0)
    return new_meas

def meas2binary(meases):
    return pd.Series(map(lambda x: 1 if x >= log_meas(500) else 0, meases))

unify_alleles = lambda x: re.sub('[-|*]', '', x)

def select_by_allele(data, allele):
    """Get data for specific allele

    Arguments:
    data - DataFrame with column "mhc"(Bdata)
    allele - allele for which we want to extract data
    """
    return data[data.mhc == allele].drop("mhc", axis = 1).reset_index(drop = True)

def select_hla(data):
    """Exract data for HLA-A, B, C, E

    Arguments:

    data9mers -  DataFrame with alleles(Bdata)
    """
    hlas = pd.Series([i[:5] for i in data.mhc])
    hla_a = hlas[hlas == "HLA-A"].index
    hla_b = hlas[hlas == "HLA-B"].index
    hla_c = hlas[hlas == "HLA-C"].index
    hla_e = hlas[hlas == "HLA-E"].index

    alleles_abce = pd.concat([data.iloc[hla_a], data.iloc[hla_b],
                             data.iloc[hla_c], data.iloc[hla_e]], axis=0).reset_index(drop=True)
    alleles_abce.mhc = [i[:5] for i in alleles_abce.mhc]

    return alleles_abce


def to_one_hot(data, length):
    """Extract OneHotEncoded representation of peptides of specific length

    Arguments:

    data - DataFrame with columns "sequence" and "peptide_length" with peptides and its length
    length - length of peptides to encode
    """
    n_mers = data[data.peptide_length == length].reset_index(drop=True)
    letters = n_mers.sequence.apply(list)
    return pd.get_dummies(pd.DataFrame(list(letters)))

def affinity_to_binary(affinity):
    """IC50 Binding affinity to binary with threshold 500 nm(from article)

    Arguments:

    affinity - Series with binding affinity to convert
    """
    return pd.Series(map(lambda x: 1 if x >= 500 else 0, affinity))

def get_kMeans_features(data, k_range):
    kMeans_meta_features = pd.DataFrame()

    for i in k_range:
        clr = KMeans(n_clusters=i, verbose=100, n_jobs=8)
        clr.fit(data)
        kMeans_meta_features[str(i)+"Means"] = clr.labels_

    return pd.get_dummies(kMeans_meta_features)
