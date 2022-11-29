import random
import numpy as np
import pandas as pd
import scipy.sparse as sparse


class Shuffling:
    def __init__(self, max_ind: int, num_shuffle: int) -> None:
        random.seed = 2911
        self.permutations = np.empty((num_shuffle, max_ind+1), dtype=np.int64)
        for n in range(num_shuffle + 1):
            iter_permutation = list(range(max_ind+1))
            random.shuffle(iter_permutation)
            self.permutations[n] = iter_permutation

    def shuffle(self, shingle_matrix: sparse.csr_matrix) -> sparse.csc_matrix:
        for n in range(self.permutations.shape[0]):
            permuted = shingle_matrix[self.permutations[n]]
            yield sparse.csc_matrix(permuted)
