

import torch


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
