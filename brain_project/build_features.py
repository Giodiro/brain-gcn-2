import scipy.io as sio
import os
import re
import numpy as np
import time

""" File to use to build the feature matrices.
"""

try:  # for local run
    data_folder = "../../data/Data-10thMay/"
except:   # for cluster
    cwd = os.getcwd()
    data_folder = "/cluster/scratch/melanibe/"+'/DATA/'

phases = {"REM_phasic": 1, "REM_tonic": 2,
          "S2_Kcomplex": 3, "S2_plain": 4,
          "S2_spindle": 5, "SWS_deep": 6
          }

subject_list = ['S01', 'S03', 'S04',
                'S05', 'S06', 'S07',
                'S08', 'S10', 'S11', 'S12']


# ------------------ LOADING DATA AND PREPARING THE MATRIX ------------------#
def prepare_X(subject_list=subject_list):
    """ Loads and merges the original MatLab files
        in one single numpy array of shape [nobs, 4095, 50].

    Args:
        subject_list: the list of subjects to load.
                      Defaults to all subjects.

    Returns:
        X: feature matrix of shape [nobs, 4095, 50]
        Y: label array of shape [nobs]
    """
    X = []
    Y = []
    i = 0
    t1 = time.time()
    for subj in os.listdir(data_folder):
        path_subj = os.path.join(data_folder, subj)
        if subj in subject_list:
            print(f"Loading subject {subj}...")
            if os.path.isdir(path_subj):
                for phase in os.listdir(path_subj):
                    path_phase = os.path.join(path_subj, phase)
                    if os.path.isdir(path_phase):
                        for file in os.listdir(path_phase):
                            if re.search(r'average', file) is not None:
                                continue
                            path_file = os.path.join(path_phase, file)
                            """
                            Other potentially interesting keys in the Matlab files:
                            RowNames: names of the 90 regions in the atlas;
                            Freqs: frequencies in Hz for each input matrix;
                            """
                            X.append(np.reshape(
                                np.asarray(sio.loadmat(path_file)['TF']),
                                          (4095, 50)))
                            Y.append(phases[phase])
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
        X_aggregated: aggregated matrix [nobs, 4095*5]
    """
    X_delta = np.mean(X[:, :, 0:4],   axis=2)  # 1 to <4 Hz
    X_theta = np.mean(X[:, :, 4:8],   axis=2)  # 4 to <8 Hz
    X_alpha = np.mean(X[:, :, 8:13],  axis=2)  # 8 - <13 Hz
    X_beta  = np.mean(X[:, :, 13:30], axis=2)  # 13 - <30 Hz
    X_gamma = np.mean(X[:, :, 30:],   axis=2)  # >=30 Hz
    X_aggregated = np.concatenate(
        (X_delta, X_theta, X_alpha, X_beta, X_gamma),
        axis=1)
    print(np.shape(X_aggregated))
    return (X_aggregated)


def transform_X_one(X):
    """ One frequency band aggregation.

    Transforms the original feature matrix by averaging
    the values over all frequency bands i.e. 'one' matrix of the report.

    Args:
        X: original feature matrix built with prepare_X

    Returns:
        X_one: aggregated matrix [nobs, 4095]
    """
    X_one = np.mean(X[:, :, :], axis=2)
    print(np.shape(X_one))
    return (X_one)


# ---------------------- build the necessary matrices ---------------------- #
if __name__ == '__main__':
    # creating the sub-directories to save the matrices if necessary
    dir = cwd+'/matrices/std/'
    if not os.path.exists(dir):
        os.makedirs(dir)
        print('Created ' + dir)
    dir = cwd+'/matrices/one/'
    if not os.path.exists(dir):
        os.makedirs(dir)
        print('Created ' + dir)
    dir = cwd+'/matrices/all/'
    if not os.path.exists(dir):
        os.makedirs(dir)
        print('Created ' + dir)
    dir = cwd+'/matrices/y/'
    if not os.path.exists(dir):
        os.makedirs(dir)
        print('Created ' + dir)
    # build and save the matrices
    for s in subject_list:
        print(s)
        X, Y = prepare_X([s])
        Xone = transform_X_one(X)
        Xstd = transform_X_std(X)
        print(np.shape(Xstd))
        np.save(cwd+'/matrices/all/{}'.format(s), X)
        np.save(cwd+'/matrices/std/{}'.format(s), Xstd)
        np.save(cwd+'/matrices/one/{}'.format(s), Xone)
        np.save(cwd+'/matrices/y/{}'.format(s), Y)
