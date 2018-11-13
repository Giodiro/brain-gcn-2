
import numpy as np
import torch

"""Utility functions for dealing with sparse tensors
"""

def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    if x.is_cuda:
        sparse_tensortype = getattr(torch.cuda.sparse, x_typename)
    else:
        sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())



def create_sparse_batch(list_of_sparse):
    batch_size = len(list_of_sparse)
    I_list = []
    V_list = []
    S_size = list_of_sparse[0].size()
    new_size = [batch_size] + list(S_size)
    for i in range(batch_size):
        I = list_of_sparse[i]._indices()
        I_size = list(I.size())
        I_size[0] = 1
        batch_idx = torch.empty(I_size, dtype=torch.long).fill_(i).to(I.device)
        I_list.append(torch.cat([batch_idx, I]))
        V_list.append(list_of_sparse[i]._values())
    new_I = torch.cat(I_list, 1)
    new_V = torch.cat(V_list, 0)
    return torch.sparse_coo_tensor(new_I, new_V, new_size)


def block_diag_from_ivs_torch(ivs_list):
    """Convert a list of sparse matrices to a block-diagonal sparse matrix.
    """
    in_nnz = []
    in_sides = []
    in_data = []; in_indices = []
    for i, v, s in ivs_list:
        in_nnz.append(v.numel())
        in_sides.append(s[0])
        in_data.append(v)
        in_indices.append(i)

    tot_nnz = sum(in_nnz)

    np_offsets = np.append(0, np.cumsum(in_sides)[:-1])
    np_offsets = np.repeat(np_offsets, repeats=in_nnz)

    out_side = np.cumsum(in_sides)[-1]
    out_shape = (out_side, out_side)

    out_data = torch.cat(in_data, 0)
    out_indices = torch.cat(in_indices, 1)

    out_indices.add_(torch.from_numpy(np_offsets).to(out_indices.device))

    return (out_indices, out_data, out_shape)



