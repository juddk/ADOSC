import torch_sparse as ts
import torch
from scipy import sparse as sps
import numpy as np


def is_hermitian(matrix):
    """Check if a matrix is Hermitian."""

    if matrix.is_sparse:
        matrix = matrix.to_dense()

    matrix_h = matrix.t().conj()

    return torch.allclose(matrix, matrix_h)


def sparse_mv(mat1, mat2):
    """
    Computes the product two matricies in pytorch.
    Maintains gradients.
    Works on matricies of any size.
    """
    mat1_indices = mat1.coalesce().indices()
    mat1_values = mat1.coalesce().values()
    mat1_shape = mat1.size()

    mat2_indices = mat2.coalesce().indices()
    mat2_values = mat2.coalesce().values()
    mat2_shape = mat2.size()

    indexC, valueC = ts.spspmm(
        mat1_indices,
        mat1_values,
        mat2_indices,
        mat2_values,
        mat1_shape[0],
        mat1_shape[1],
        mat2_shape[1],
        coalesced=True,
    )

    return torch.sparse_coo_tensor(indexC, valueC, (mat1_shape[0], mat2_shape[1]))


def sps_to_torch_sparse(matrix):
    """Converts a scipy.sparse matrix to a PyTorch sparse tensor."""
    if not sps.isspmatrix_coo(matrix):
        matrix = matrix.tocoo()

    values = matrix.data
    indices = np.vstack((matrix.row, matrix.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = matrix.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def torch_sparse_to_sps(tensor):
    """Converts a torch.sparse object to a scipy.sparse object."""

    tensor = tensor.coalesce()
    values = tensor.values().numpy()
    indices = tensor.indices().numpy()
    shape = tensor.size()

    return sps.coo_matrix((values, (indices[0], indices[1])), shape=shape)


def torch_sparse_kron(A: torch.Tensor, B: torch.Tensor):
    """
    Compute the Kronecker product of two sparse tensors.

    Parameters:
        A (torch.sparse.FloatTensor): First sparse tensor.
        B (torch.sparse.FloatTensor): Second sparse tensor.

    Returns:
        torch.sparse.FloatTensor: The Kronecker product of A and B.
    """

    A_indices = A._indices()
    A_values = A._values()

    B_indices = B._indices()
    B_values = B._values()

    m, n = A_indices.size(1), B_indices.size(1)
    rows = (A_indices[0, :, None] * B.size(0) + B_indices[0]).view(-1)
    cols = (A_indices[1, :, None] * B.size(1) + B_indices[1]).view(-1)
    kronecker_indices = torch.stack((rows, cols))

    kronecker_values = (A_values[:, None] * B_values).flatten()

    size = (A.size(0) * B.size(0), A.size(1) * B.size(1))
    result = torch.sparse.FloatTensor(kronecker_indices, kronecker_values, size)

    return result.coalesce()
