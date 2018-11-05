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


def kl_categorical(preds, log_prior, num_atoms, eps=1e-16):
    kl_div = preds * (torch.log(preds + eps) - log_prior)
    return kl_div.sum() / (num_atoms * preds.size(0))
