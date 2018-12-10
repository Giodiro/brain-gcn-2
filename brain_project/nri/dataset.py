import os
import json
import numpy as np
from collections import OrderedDict
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from cache import LRUCache
from util import time_str

__all__ = [
    "ALL_SUBJECTS",
    "ALL_PHASES",
    "CLASS_NAMES",
    "EEGDataset1",
    "EEGDataset2",
    "SyntheticDataset",
    "collate_tensors",
]


ALL_SUBJECTS = ['S01', 'S03', 'S04',
                'S05', 'S06', 'S07',
                'S08', 'S10', 'S11',
                'S12']

ALL_PHASES = OrderedDict([
                ("REM_phasic", 0),
                ("REM_tonic", 1),
                ("S2_Kcomplex", 2),
                ("S2_plain", 3),
                ("S2_spindle", 4),
                ("SWS_deep", 5)
                ])

CLASS_NAMES = list(ALL_PHASES.keys())


class EEGDataset1(Dataset):
    """PyTorch dataloader for the imaginary coherence dataloader (dataset 1).
    """

    all_transformations = ["one", "std"]


    def __init__(self, data_folder, file_indices, subj_data, transformation="none", super_node=False):
        """
        Args:
          data_folder : str
            The root folder of the preprocessed data.
          file_indices : Dict[int->int]
            Converts linear indices useful to iterate through the dataset
            into keys for the `subj_data` structure.
          subj_data : dict
            Information about the file location of each sample
        """
        super(EEGDataset1, self).__init__()
        if transformation not in all_transformations:
            raise ValueError(f"Transformation must be in {all_transformations}.")

        self.super_node = super_node
        self.transformation = transformation
        self.data_folder = data_folder
        self.num_nodes = 90

        self.xfile_cache = LRUCache(capacity=50)
        self.yfile_cache = LRUCache(capacity=500)
        self.subj_data = subj_data
        self.file_indices = file_indices

    def get_xy_files(self, idx):
        idx = self.file_indices[idx]
        sdata = self.subj_data[idx]
        x_file = os.path.join(self.data_folder, "X", sdata["file"])
        y_file = os.path.join(self.data_folder, "Y", sdata["file"])
        iif = sdata["index_in_file"]

        return x_file, y_file, iif

    def __getitem__(self, idx):
        x_file, y_file, iif = self.get_xy_files(idx)

        X = self.xfile_cache.load(x_file, iif) # [4095, 50]
        X = self.transform(X) # [num_freq, 90, 90]
        Y = self.yfile_cache.load(y_file, iif)

        sample = {
            "X": torch.tensor(X, dtype=torch.float32),
            "Y": torch.tensor(Y, dtype=torch.long),
        }
        return sample

    def transform(self, X):
        """
        Args:
         X : numpy array [4095, 50]
        Returns:
         X_transformed : numpy array [num_freq, 90, 90]
        """
        if self.transformation == "std":
            X_delta = np.mean(X[:, 0:4],   axis=-1)  # 1 to <4 Hz
            X_theta = np.mean(X[:, 4:8],   axis=-1)  # 4 to <8 Hz
            X_alpha = np.mean(X[:, 8:13],  axis=-1)  # 8 - <13 Hz
            X_beta  = np.mean(X[:, 13:30], axis=-1)  # 13 - <30 Hz
            X_gamma = np.mean(X[:, 30:],   axis=-1)  # >=30 Hz
            X_aggregated = np.stack(
                (X_delta, X_theta, X_alpha, X_beta, X_gamma),
                axis=1)
        elif self.transformation == "one":
            X_aggregated = np.mean(X, axis=-1).expand_dims(1)

        As = []
        for band in range(X.shape[1]):
            A = self.adj_from_tril(X_aggregated[:,band],
                                   num_nodes=self.num_nodes,
                                   super_node=self.super_node) # 90 x 90
            As.append(A)
        A = np.stack(As, axis=0).astype(np.float32) # num_freq x 90 x 90
        return A

    def __len__(self):
        return len(self.file_indices)

    @property
    def num_bands(self):
        if self.transformation == "std":
            return 5
        elif self.transformation == "one":
            return 1

    def adj_from_tril(self, one_coh_arr):
        """ builds the A hat matrix of the paper for one sample.
        https://github.com/brainstorm-tools/brainstorm3/blob/master/toolbox/process/functions/process_compress_sym.m shows that
        the flat matrix contains the lower triangular values of the initial symmetric matrix.

        Args:
          one_coh_arr : array [num_nodes*(num_nodes+1)/2]
          super_node : bool (default False)

        Returns:
          A : array [num_nodes, num_nodes]
        """
        # First construct weighted adjacency matrix
        A = np.zeros((self.num_nodes,self.num_nodes))
        index = np.tril_indices(self.num_nodes)
        A[index] = one_coh_arr
        A = (A + A.T)
        if self.super_node:
            A = np.concatenate((A, np.ones((self.num_nodes, 1))), axis = 1) # adding the super node
            A = np.concatenate((A, np.ones((1, self.num_nodes+1))), axis = 0)
        # A tilde from the paper
        di = np.diag_indices(self.num_nodes)
        A[di] = A[di]/2
        A_tilde = A + np.eye(self.num_nodes)
        # D tilde power -0.5
        D_tilde_inv = np.diag(np.power(np.sum(A_tilde, axis=0), -0.5))
        # Finally build A_hat
        A_hat = np.matmul(D_tilde_inv, np.matmul(A_tilde, D_tilde_inv))
        return A_hat



class EEGDataset2(Dataset):
    """PyTorch dataloader for the temporal sequence dataset (dataset 2).
    This dataset has 435 identified ROIs, each containing the mean activation
    of sources in the regions at every time-point. The data is organized by
    sleep activity. Each file contains activations for the ROIs at 1500
    time points (at 1Hz?) .

    Note:
      The dataset has very small values (i.e. 1e-10 range). This may cause
      precision errors when using single-precision floating point numbers.
      This class offers two normalization options:
       - standardizing each ROI to 0-mean, unit variance (requires preprocessing
         the whole dataset to extract global statistics)
       - scaling by a large value (NORM_CONSTANT).
    """

    all_normalizations = ["standard",     # Standardize each ROI
                          "none",         # Multiply all values by NORM_CONSTANT
                          "val",          # Indicates that this is a validation loader so normalization is loaded from the tr loader
                         ]

    NORM_CONSTANT = 1.0e10

    def __init__(self, data_folder, file_indices, subj_data, normalization="none"):
        """
        Args:
          data_folder : str
            The root folder of the preprocessed data.
          file_indices : Dict[int->int]
            Converts linear indices useful to iterate through the dataset
            into keys for the `subj_data` structure.
          subj_data : dict
            Information about the file location of each sample
          normalization : str
            The type of normalization to use for the data. This can be either
            standard, none or val. val should only be used if this is a
            validation dataset and the statistics are extracted from the
            training set.
        """
        super(EEGDataset2, self).__init__()
        if normalization not in EEGDataset2.all_normalizations:
            raise ValueError(f"Normalization must be in {all_normalizations}.")

        self.normalization = normalization
        self.data_folder = data_folder

        self.xfile_cache = LRUCache(capacity=50)
        self.yfile_cache = LRUCache(capacity=500)
        self.subj_data = subj_data
        self.file_indices = file_indices

        self.init_normalizer()

    def get_xy_files(self, idx):
        idx = self.file_indices[idx]
        sdata = self.subj_data[idx]
        x_file = os.path.join(self.data_folder, "X", sdata["file"])
        y_file = os.path.join(self.data_folder, "Y", sdata["file"])
        iif = sdata["index_in_file"]

        return x_file, y_file, iif

    def __getitem__(self, idx):
        x_file, y_file, iif = self.get_xy_files(idx)

        X = self.xfile_cache.load(x_file, iif) # [num_nodes, time_steps]
        X = self.normalize(X)
        Y = self.yfile_cache.load(y_file, iif)

        sample = {
            "X": torch.tensor(X, dtype=torch.float32),
            "Y": torch.tensor(Y, dtype=torch.long),
        }
        return sample

    def __len__(self):
        return len(self.file_indices)

    def init_normalizer(self):
        if self.normalization == "val":
            return

        print(f"{time_str()} Initializing normalization ({self.normalization}) statistics.")
        if self.normalization == "none":
            self.scaler = None
            return

        self.scaler = StandardScaler(copy=False, with_mean=True, with_std=True)
        # Iterate all samples to compute statistics.
        # TODO: This can be optimized to feed the scalers all samples read from a file
        #       but care must be taken to actually only feed it samples whose id is in
        #       the allowed ids.
        for i in range(len(self)):
            x_file, y_file, iif = self.get_xy_files(i)
            X = self.xfile_cache.load(x_file, iif)
            self.scaler.partial_fit(X)

    def normalize(self, data):
        """
        Args:
         - data : array [423, time_steps]

        Returns:
         - norm_data : array [423, time_steps]
        """
        if self.normalization == "val":
            raise ValueError("Normalization cannot be `val`, must be set to a concrete value.")

        if self.normalization == "none":
            data = data * EEGDataset2.NORM_CONSTANT
        else:
            data = self.scaler.transform(data)

        return data.astype(np.float32)



class SyntheticDataset(Dataset):
    """PyTorch data-loader for a synthetic temporal dataset.
    This is designed to simulate dataset2, so will have the same kind of interface.
    """

    all_normalizations = ["standard",     # Standardize each ROI
                          "none",         # Multiply all values by NORM_CONSTANT
                          "val",          # Indicates that this is a validation loader so normalization is loaded from the tr loader
                          ]

    def __init__(self, samples, sample_labels, normalization="none"):
        super(SyntheticDataset, self).__init__()
        if normalization not in SyntheticDataset.all_normalizations:
            raise ValueError(f"Normalization must be in {all_normalizations}.")

        self.normalization = normalization
        self.samples = samples
        self.sample_labels = sample_labels

        self.init_normalizer()

    def __getitem__(self, idx):
        X = self.normalize(self.samples[idx])
        Y = self.sample_labels[idx]

        sample = {
            "X": torch.tensor(X, dtype=torch.float32),
            "Y": torch.tensor(Y, dtype=torch.long),
        }

        return sample

    def init_normalizer(self):
        if self.normalization == "val":
            return

        print(f"{time_str()} Initializing normalization ({self.normalization}) statistics.")
        if self.normalization == "none":
            self.scaler = None
            return

        self.scaler = StandardScaler(copy=False, with_mean=True, with_std=True)
        self.scaler.fit(np.concatenate(self.samples, 0))

    def normalize(self, data):
        """
        Args:
         - data : array [num_nodes, time_steps]

        Returns:
         - norm_data : array [num_nodes, time_steps]
        """
        if self.normalization == "val":
            raise ValueError("Normalization cannot be `val`, must be set to a concrete value.")

        if self.normalization != "none":
            data = self.scaler.transform(data)

        return data.astype(np.float32)

    def __len__(self):
        return len(self.samples)


def collate_tensors(tensors):
    batch = {}

    keys = list(tensors[0].keys())
    for k in keys:
        collate_v = torch.stack([t[k] for t in tensors], 0)
        batch[k] = collate_v

    return batch
