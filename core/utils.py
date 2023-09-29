import torch_sparse as ts
import torch


def sparse_mv(mat1, mat2):
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
