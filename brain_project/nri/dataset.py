import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn

from cache import LRUCache


all_subjects = ['S01', 'S03', 'S04',
                'S05', 'S06', 'S07',
                'S08', 'S10', 'S11', 'S12']

all_phases = {"REM_phasic": 1, "REM_tonic": 2,
              "S2_Kcomplex": 3, "S2_plain": 4,
              "S2_spindle": 5, "SWS_deep": 6
             }


class EEG_Dataset1(Dataset):
    """PyTorch dataloader for the imaginary coherence dataloader (dataset 1).
    """

    all_transformations = ["one", "std"]

    def __init__(self, data_folder, subject_list=all_subjects, transformation="std", super_node=False):
        if transformation not in all_transformations:
            raise ValueError(f"Transformation must be in {all_transformations}.")
        if len(subject_list) == 0:
            raise ValueError(f"Must specify at least one subject.")

        self.super_node = super_node
        self.num_nodes = 90

        self.X, self.Y = self.prepare_X(data_folder, subject_list)
        self.X = self.X.astype(np.float32)
        self.Y = self.Y.astype(np.int)

        # Transform data by aggregating the 50 frequencies
        if transformation == "std":
            self.X = transform_X_std(self.X)
        elif transformation == "one":
            self.X = transform_X_one(self.X)

    def __getitem__(self, idx):
        sample = {}

        for band in range(self.num_bands):
            Ai = build_onegraph_A(self.X[idx, :, band],
                                  num_nodes=self.num_nodes,
                                  super_node=self.super_node)
            sample[f"A{band}"] = torch.from_numpy(Ai.astype(np.float32))

        sample["Y"] = torch.tensor(self.Y[idx], dtype=torch.long)

        return sample

    def __len__(self):
        return len(self.Y)

    @property
    def num_bands(self):
        return self.X.shape[2]

    @staticmethod
    def collate(tensors):
        batch = {}

        keys = list(tensors[0].keys())
        for k in keys:
            collate_v = torch.stack([t[k] for t in tensors], 0)
            batch[k] = collate_v

        return batch

    def prepare_X(self, data_folder, subject_list):
        """ Loads and merges the original MatLab files
            in one single numpy array of shape [nobs, 4095, 50].

        Args:
            subject_list: the list of subjects to load.
                          Defaults to all subjects.

        Returns:
            X: feature matrix of shape [nobs, 4095, 50]
            Y: label array of shape [nobs]
        """
        X = []; Y = []
        i = 0
        t1 = time.time()
        for subj in os.listdir(data_folder):
            path_subj = os.path.join(data_folder, subj)
            if subj not in subject_list:
                continue
            if not os.path.isdir(path_subj):
                continue
            print(f"Loading subject {subj}...")
            for phase in os.listdir(path_subj):
                path_phase = os.path.join(path_subj, phase)
                if not os.path.isdir(path_phase):
                    continue
                if phase not in all_phases:
                    continue
                for file in os.listdir(path_phase):
                    if re.search(r'average', file) is not None:
                        continue
                    path_file = os.path.join(path_phase, file)
                    """
                    Other potentially interesting keys in the Matlab files:
                    RowNames: names of the 90 regions in the atlas;
                    Freqs: frequencies in Hz for each input matrix;
                    """
                    mat_data = sio.loadmat(path_file)['TF']
                    np_data = np.asarray(mat_data).reshape(4095, 50)
                    X.append(np_data)
                    Y.append(all_phases[phase])
                    i += 1
        t2 = time.time()
        X = np.asarray(X)
        Y = np.asarray(Y)
        print(f"Loaded {i} observations of shape {X.shape} in {t2 - t1:.2f}s.")

        return (X, Y)

    def transform_X_std(X):
        """ Standard frequency band aggregation.

        Transforms the original feature matrix by aggregating
        the values per standard frequency band i.e. 'std' matrix of the report.

        Args:
            X: original feature matrix built with prepare_X

        Returns:
            X_aggregated: aggregated matrix [nobs, 4095, 5]
        """
        X_delta = np.mean(X[:, :, 0:4],   axis=2)  # 1 to <4 Hz
        X_theta = np.mean(X[:, :, 4:8],   axis=2)  # 4 to <8 Hz
        X_alpha = np.mean(X[:, :, 8:13],  axis=2)  # 8 - <13 Hz
        X_beta  = np.mean(X[:, :, 13:30], axis=2)  # 13 - <30 Hz
        X_gamma = np.mean(X[:, :, 30:],   axis=2)  # >=30 Hz
        X_aggregated = np.stack(
            (X_delta, X_theta, X_alpha, X_beta, X_gamma),
            axis=2)
        return X_aggregated

    def transform_X_one(X):
        """Single frequency band aggregation.

        Transforms the original feature matrix by averaging
        the values over all frequency bands i.e. 'one' matrix of the report.

        Args:
            X: original feature matrix built with prepare_X

        Returns:
            X_one: aggregated matrix [nobs, 4095, 1]
        """
        X_one = np.mean(X[:, :, :], axis=2).expand_dims(2)
        return X_one


def build_onegraph_A(one_coh_arr, num_nodes, super_node=False):
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
    A = np.zeros((num_nodes,num_nodes))
    index = np.tril_indices(num_nodes)
    A[index] = one_coh_arr
    A = (A + A.T)
    if super_node:
        A = np.concatenate((A, np.ones((num_nodes, 1))), axis = 1) # adding the super node
        A = np.concatenate((A, np.ones((1, num_nodes+1))), axis = 0)
    # A tilde from the paper
    di = np.diag_indices(num_nodes)
    A[di] = A[di]/2
    A_tilde = A + np.eye(num_nodes)
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
      We therefore need to investigate normalization (0-mean, 1-var) options:
       - spatial normalization may lose information if certain time frames
         have less overall activity
       - temporal normalization may lose information if certain ROIs have
         lower/higher mean activity.
      Otherwise one could just multiply the values by a suitably large
      (e.g. 1e-9) number to avoid precision issues.
    """

    all_normalizations = ["spat-mean",     # Make each time-frame 0 mean
                          "spat-standard", # Standardize each time-frame
                          "temp-mean",     # Make each ROI 0 mean
                          "temp-standard", # Standardize each ROI
                          "none",          # Multiply all values by 1e9
                          ]

    def __init__(self, data_folder, file_indices, subj_data, normalization="none"):
        """
        Args:
          data_folder : str
          subject_list : List[str]
          normalization : str
          num_timepoints : int
            The number time-points to use in each sample. Each file currently
            holds 2501 time points, so this is the maximum (although this
            limit may change if we get full trajectories). Lower values will
            cause each file to be split up into multiple samples (and possibly
            some of the last frames will be discarded).
        """
        super(EEGDataset2, self).__init__()
        if normalization not in EEGDataset2.all_normalizations:
            raise ValueError(f"Normalization must be in {all_normalizations}.")

        self.tot_tpoints = 2501
        self.num_nodes = 423
        self.normalization = normalization
        self.data_folder = data_folder

        self.xfile_cache = LRUCache(capacity=50)
        self.yfile_cache = LRUCache(capacity=500)
        self.subj_data = subj_data

        self.file_indices = file_indices

    def __getitem__(self, idx):
        idx = self.file_indices[idx]
        sdata = self.subj_data[idx]
        x_file = os.path.join(self.data_folder, "X", sdata["file"])
        y_file = os.path.join(self.data_folder, "Y", sdata["file"])
        iif = sdata["index_in_file"]

        X = self.xfile_cache.load(x_file, iif)
        Y = self.yfile_cache.load(y_file, iif)

        # TODO: Run normalization

        sample = {
            "X": torch.tensor(X.astype(np.float32).transpose()),
            "Y": torch.tensor(Y, dtype=torch.long),
        }
        return sample

    def __len__(self):
        return len(self.file_indices)

    @staticmethod
    def collate(tensors):
        batch = {}

        keys = list(tensors[0].keys())
        for k in keys:
            collate_v = torch.stack([t[k] for t in tensors], 0)
            batch[k] = collate_v

        return batch



