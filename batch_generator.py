# From https://github.com/fchollet/keras/issues/1638
import threading

import numpy as np
from numpy.random import randint


# class threadsafe_iter:
#     """Takes an iterator/generator and makes it thread-safe by
#     serializing call to the `next` method of given iterator/generator.
#     """
#     def __init__(self, it):
#         self.it = it
#         self.lock = threading.Lock()

#     def __iter__(self):
#         return self

#     def __next__(self):
#         with self.lock:
#             return next(self.it)
        
#     def next(self):
#         with self.lock:
#             return self.it.next()


# def threadsafe_generator(f):
#     """A decorator that takes a generator function and makes it thread-safe.
#     """
#     def g(*a, **kw):
#         return threadsafe_iter(f(*a, **kw))
#     return g


# @threadsafe_generator
def generate_batch_imba(X_list, y, batch_size):
    while True:
        sampled_indices = randint(0, X_list[0].shape[0], size=batch_size)
        yield [X_list[0][sampled_indices], X_list[1][sampled_indices]], y[sampled_indices]

            
# @threadsafe_generator
def generate_batch_balanced(X_list, y, batch_size, indices_strong, indices_weak):
    while True:
        to_sample_strong = batch_size // 2
        to_sample_weak   = batch_size // 2
        sampled_indices_strong = indices_strong[randint(0, indices_strong.shape[0], size=to_sample_strong)]
        sampled_indices_weak   = indices_weak[randint(0, indices_weak.shape[0], size=to_sample_weak)]
        yield [np.vstack([X_list[0][sampled_indices_strong], X_list[0][sampled_indices_weak]]), \
               np.vstack([X_list[1][sampled_indices_strong], X_list[1][sampled_indices_weak]])], \
              np.vstack([y[sampled_indices_strong], y[sampled_indices_weak]])

            
# @threadsafe_generator
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
            

# @threadsafe_generator   
def generate_batch_weighted(X_list, y, batch_size, indices_strong, indices_weak, weights_train):
    while True:
        to_sample_strong = int(batch_size / 2)
        to_sample_weak   = int(batch_size / 2)
        sampled_indices_strong = indices_strong[randint(0, indices_strong.shape[0], size=to_sample_strong)]
        sampled_indices_weak   = indices_weak[randint(0, indices_weak.shape[0], size=to_sample_weak)]
        yield [np.vstack([X_list[0][sampled_indices_strong], X_list[0][sampled_indices_weak]]), \
               np.vstack([X_list[1][sampled_indices_strong], X_list[1][sampled_indices_weak]])], \
              np.vstack([y[sampled_indices_strong], y[sampled_indices_weak]]), \
              np.vstack([weights_train[sampled_indices_strong], weights_train[sampled_indices_weak]]).reshape((batch_size,))