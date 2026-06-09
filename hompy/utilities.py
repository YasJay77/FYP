import numpy as np
import scipy.sparse as ss


def take_sparse(k, c):
    """principal submatrix k[c,:][:,c] as CSR; sparse fancy-indexing to keep memory down"""
    c_arr = np.asarray(c, dtype=np.intp)
    # row-slice then column-slice, both O(nnz)
    return ss.csr_matrix(k[c_arr, :][:, c_arr])
