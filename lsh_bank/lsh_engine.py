import random
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from lsh_bank import Shingling


class Shuffling:
    """
    Class used to permute the first axis of the shingle matrix. This is a necessary step in order to compute the
    minhash signatures for an LSH implementation. The object is initialized by receiving the number of features in the
    shingle matrix and the number of permutations.
    """

    def __init__(self, num_feature: int, num_shuffle: int) -> None:
        # Set a seed for reproducibility of the permutations
        random.seed = 2911
        # Initialize the array that holds the indexes for the permutations
        self.permutations = np.empty((num_shuffle, num_feature), dtype=np.int32)
        # Initialize list that will be shuffled
        iter_permutation = list(range(num_feature))
        for n in range(num_shuffle):
            # Shuffle
            random.shuffle(iter_permutation)
            # Set the values for a row of the Numpy array
            self.permutations[n] = iter_permutation

    def shuffle(self, shingle_matrix: sparse.csr_matrix) -> sparse.csc_matrix:
        """
        Generator that performs the actual repeated shuffling in an iterative fashion.
        :param shingle_matrix: The sparse shingle matrix whose first axis we are permuting over.
        :returns Generator object yielding a permuted shingle matrix at each iteration.
        """
        for n in range(self.permutations.shape[0]):
            # Fancy indexing of the sparse matrix
            permuted = shingle_matrix[self.permutations[n]]
            # Return sparse column matrix since with argmax for the MinHashing we will be accessing the columns
            # (column slicing) and not the rows.
            yield sparse.csc_matrix(permuted)


def min_hashing(shuffling_obj: Shuffling, shingle_matrix: sparse.csr_matrix):
    """
    This function takes care of the MinHashing, using other objects defined in this package.
    :param shuffling_obj: Object of type Shuffling, whose shuffle method is used to perform the repeated shuffling.
    :param shingle_matrix: shingle matrix to shuffle from.
    :return: The dense Numpy array with the minhash signatures.
    """
    # Allocate the Numpy matrix that holds the minhash signatures
    minhash_signatures = np.empty((shuffling_obj.permutations.shape[0], shingle_matrix.shape[1]),
                                  dtype=np.int32)
    # Loop over the customers and add the rows to the minhash signatures Numpy Array
    for idx, customer in enumerate(shuffling_obj.shuffle(shingle_matrix)):
        minhash_signatures[idx] = customer.argmax(axis=0)

    return minhash_signatures


def lsh(minhash_signatures_dataset: np.array, minhash_signature_query: np.array, customers_dataset: pd.DataFrame,
        n_bands: int) -> pd.DataFrame:
    """

    :param minhash_signatures_dataset:
    :param minhash_signature_query:
    :param customers_dataset:
    :param n_bands:
    :return:
    """
    # "Collapse the length of the array" onto a third axis
    collapsed_signatures_dataset = \
        minhash_signatures_dataset.reshape((-1, minhash_signatures_dataset.shape[1], n_bands))

    # Collapse the query signature the same way
    collapsed_query = minhash_signature_query.reshape(-1, 1, n_bands)

    # Use broadcasting
    matches = np.all(collapsed_signatures_dataset == collapsed_query, axis=2).sum(axis=0)
    print(len(np.nonzero(matches)[0]))
    similar = customers_dataset.assign(matches=matches)
    similar = similar[similar.matches != 0]

    return similar


if __name__ == '__main__':
    import pickle
    import dotenv
    import os

    dotenv.load_dotenv('../../ext_variables.env')
    with open(os.path.join(os.getenv("PATH_FILES_ADM"), 'shingling_obj.pickle'), 'rb') as file:
        shingling_obj = pickle.load(file)
    shuffling_obj = Shuffling(shingling_obj.shingle_matrix.shape[0], 400)
    # signatures_array = min_hashing(shuffling_obj, shingling_obj.shingle_matrix)

    with open(os.path.join(os.getenv("PATH_FILES_ADM"), 'signatures_array.pickle'), 'rb') as file:
        signatures_array = pickle.load(file)

    query_array = pd.read_csv('../query_users.csv')
    query_array_transformed = shingling_obj.transform(query_array)
    query_array_signatures = min_hashing(shuffling_obj, query_array_transformed)
    for query in query_array_signatures.T:
        test = lsh(signatures_array, query, shingling_obj.consumer_table, 2)
        print(test)
        break



