import os, errno
from datetime import datetime
import time
import itertools

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


def plot_confusion_matrix(ytrue, ypred, class_names, normalize=True):
    from sklearn import metrics
    import matplotlib.pyplot as plt

    cnf_mat = metrics.confusion_matrix(ytrue, ypred)

    if normalize:
        cnf_mat = cnf_mat.astype('float') / cnf_mat.sum(axis=1)[:, np.newaxis]

    np.set_printoptions(precision=2)
    plt.figure()

    plt.imshow(cnf_mat, interpolation="nearest", cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    fmt = '.2f' if normalize else 'd'
    thresh = cnf_mat.max() / 2.
    for i, j in itertools.product(range(cnf_mat.shape[0]), range(cnf_mat.shape[1])):
        plt.text(j, i, format(cnf_mat[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cnf_mat[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True')
    plt.xlabel('Predicted')

    return plt.gcf()
