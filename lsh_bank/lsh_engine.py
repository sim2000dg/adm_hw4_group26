import random
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from lsh_bank import Shingling
from collections import defaultdict, Counter
from itertools import chain


class Shuffling:
    """
    Class used to permute the first axis of the shingle matrix. This is a necessary step in order to compute the
    minhash signatures for an LSH implementation. The object is initialized by receiving the number of features in the
    shingle matrix, the number of permutations and a seed.
    """
    def __init__(self, num_feature: int, num_shuffle: int, seed: int) -> None:
        # Set a seed for reproducibility of the permutations
        random.seed(seed)
        # Initialize the array that holds the indexes for the permutations
        self.permutations = np.empty((num_shuffle, num_feature), dtype=np.int32)
        # Initialize list that will be shuffled
        iter_permutation = list(range(num_feature))
        for n in range(num_shuffle):
            # Shuffle
            random.shuffle(iter_permutation)
            # Set the values for a row of the Numpy array
            self.permutations[n] = iter_permutation

    def shuffle(self, shingle_matrix: np.array):
        """
        Generator that performs the actual repeated shuffling in an iterative fashion.
        :param shingle_matrix: The shingle matrix whose first axis we are permuting over.
        :returns Generator object yielding a permuted shingle matrix at each iteration.
        """
        for n in range(self.permutations.shape[0]):
            # Fancy indexing of the shingle matrix
            permuted = shingle_matrix[self.permutations[n]]
            yield permuted


def min_hashing(shuffling_obj: Shuffling, shingle_matrix: sparse.csc_matrix) -> np.array:
    """
    This function takes care of the MinHashing, using other objects defined in this package.
    :param shuffling_obj: Object of type Shuffling, whose shuffle method is used to perform the repeated shuffling.
    :param shingle_matrix: shingle matrix to shuffle from.
    :return: The dense Numpy array with the minhash signatures.
    """
    shingle_matrix = shingle_matrix.toarray()

    # Allocate the Numpy matrix that holds the minhash signatures
    minhash_signatures = np.empty((shuffling_obj.permutations.shape[0], shingle_matrix.shape[1]),
                                  dtype=np.int32)
    # Loop over the shuffles and add the rows to the minhash signatures Numpy Array
    for idx, iteration in enumerate(shuffling_obj.shuffle(shingle_matrix)):
        minhash_signatures[idx] = iteration.argmax(axis=0)

    return minhash_signatures


def lsh_buckets(minhash_signatures_dataset: np.array, n_bands: int) -> dict:
    """
    This functions returns the lsh buckets (a Python dictionary) for a given matrix of minhash signatures, depending on
    the number of bands chosen.
    :param minhash_signatures_dataset: The matrix of the minhash signatures.
    :param n_bands: The number of bands.
    :return: The dictionary containing the buckets.
    """
    buckets = defaultdict(list)  # Default dict is useful to avoid try and except clauses
    # residual to handle cases where n_bands is not a divisor of the length of the signature matrix on the first axis
    residual = minhash_signatures_dataset.shape[0] % n_bands
    if residual:  # If residual is not zero, the residual part is considered as a separate band
        minhash_signatures_dataset, residual = \
            (minhash_signatures_dataset[:minhash_signatures_dataset.shape[0] - residual],
             minhash_signatures_dataset[minhash_signatures_dataset.shape[0] - residual:])
    for idx_cust, column in enumerate(minhash_signatures_dataset.T):  # Iterate over the columns/the customers
        # iterate over the bands
        for idx_band, band in enumerate(chain(np.split(column, n_bands), [residual]) if residual else
                                        np.split(column, n_bands)):
            # each bucket has as key a minhash signature specific to the band and the index for the band itself
            # the key for a dictionary needs to be hashable, and a tuple of integers is hashable, since
            # the tuple is immutable and integers are too
            buckets[tuple([idx_band] + band.tolist())].append(idx_cust)

    return buckets
