import os, errno
from datetime import datetime
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def time_str():
    now = datetime.now()
    return now.strftime("[%m-%d %H:%M:%S]")


def safe_time_str():
    now = datetime.now()
    return now.strftime("%m.%d.%H.%M.%S")


def list_to_safe_str(l):
    return str(l).replace(" ", "") \
                 .replace("[", "") \
                 .replace("]", "") \
                 .replace(",", ".")


def mkdir_p(path):
    """Recursively create directories."""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def sample_gumbel(shape, is_cpu, eps=1e-10):
    """Sample from Gumbel(0, 1)
    """
    if is_cpu:
        U = torch.FloatTensor(shape).normal_()
    else:
        U = torch.cuda.FloatTensor(shape).normal_()

    return - torch.log(eps - torch.log(U + eps))


def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """Draw a sample from the Gumbel-Softmax distribution
    """
    is_cpu = logits.device == "cpu"
    gumbel_noise = sample_gumbel(logits.size(), is_cpu, eps=eps)
    y = logits + gumbel_noise
    return F.softmax(y / tau, dim=-1)


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.

    Args:
     logits : tensor [batch_size, n_class]
        unnormalized log-probs
     tau : float (default 1)
        non-negative scalar temperature
     hard : bool (default False)
        if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes
    Constraints:
     - this implementation only works on batch_size x num_features tensor for now

    NOTE:
     Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3
     https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
     (MIT license)
    """
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = torch.zeros(*shape, device=logits.device)
        y_hard = y_hard.scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = (y_hard - y_soft.data) + y_soft # TODO: test this works
    else:
        y = y_soft

    return y


def kl_categorical(preds, log_prior, num_atoms, eps=1e-16):
    kl_div = preds * (torch.log(preds + eps) - log_prior)
    return kl_div.sum() / (num_atoms * preds.size(0))
