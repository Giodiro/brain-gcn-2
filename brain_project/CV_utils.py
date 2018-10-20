import numpy as np
import pandas as pd
import os

from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
cwd = os.getcwd()

""" This file defines all the customized
cross validations used in the experiments.
"""


# ---------------------- Upsampling Kfold ---------------------- #
class UpsampleStratifiedKFold:
    """ Generates stratified K-Fold splits with upsampling
    to have balanced class observations in each split.
    Replicating the structure of the StratifiedKFold
    class in sklearn.
    """
    def __init__(self, n_splits=3):
        self.n_splits = n_splits

    def split(self, X, y, groups=None):
        skf = StratifiedKFold(n_splits=self.n_splits,
                              shuffle=True,
                              random_state=42)
        for train_idx, test_idx in skf.split(X, y):
            neg_ix = np.where([y[i] == 0 for i in train_idx])[0]
            neg_ix = [train_idx[i] for i in neg_ix]
            pos_ix = np.where([y[i] == 1 for i in train_idx])[0]
            aug_pos_ix = np.random.choice(pos_ix, size=len(neg_ix),
                                          replace=True)
            aug_pos_ix = [train_idx[i] for i in aug_pos_ix]
            train_idx = np.append(neg_ix, aug_pos_ix)
            yield train_idx, test_idx

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits


# ---------------------- ACROSS CV ---------------------- #
def AcrossSubjectCV(estimator, logger, subject_list,
                    mat='std', upsample=False):
    """ Implements custom across subject CV.
    Note:
        Training on all the subjects except one.
        Testing on the hold-on subject.
        nb_subject-fold cross validation.

    Warning:
        - Estimator has to be child BaseEstimator class
        from sklearn.
        - If you use GCN estimator the type of feature matrix
        has to be standard frequency band aggregation i.e.
        type std.
        - Assumes that all the matrices are already prepared
        per subject (see Readme)

    Args:
        estimator: estimator to use
        logger: logger to print the results to.
        subject_list: list of str with the subject names
                      to include in the CV.
        mat: type of feature matrix.
        upsample: whether to use upsampling or not.

    Returns:
        results: dataframe containing the
                 results averaged over all folds
        metrics: dataframe containing the
                 results per fold
        confusion: a list of confusion matrix for each fold
        conf_perc: a list of percentage confusion matrix for each fold
    """
    roc_auc = []
    accuracyCV = []
    confusion = []
    conf_perc = []
    bal_acc = []
    for s in subject_list:
        X_test = np.load(cwd+'/matrices/{}/{}.npy'.format(mat, s))
        Y_test = np.load(cwd+'/matrices/y/{}.npy'.format(s))
        # reduce to 2 classes
        Y_test = [1 if ((y == 1) or (y == 2)) else 0 for y in Y_test]
        print("Preparing fold only {}".format(s))
        print("Preparing fold except {}".format(s))
        first = True
        for other in subject_list:
            if not other == s:
                print(other)
                if first:
                    X_train = np.load(cwd+'/matrices/{}/{}.npy'
                                      .format(mat, other))
                    Y_train = np.load(cwd+'/matrices/y/{}.npy'
                                      .format(other))
                    first = False
                else:
                    X_train = np.concatenate(
                       (X_train,
                        np.load(cwd+'/matrices/{}/{}.npy'.format(mat, other))),
                       axis=0)
                    Y_train = np.concatenate(
                       (Y_train,
                        np.load(cwd+'/matrices/y/{}.npy'.format(other))),
                       axis=0)
        Y_train = [1 if ((y == 1) or (y == 2)) else 0 for y in Y_train]
        n = len(X_train)
        print(n)
        print(len(X_test))
        print(n+len(X_test))
        if upsample:   # perform upsampling only on training fold
            neg_ix = np.where([Y_train[i] == 0 for i in range(n)])[0]
            pos_ix = np.where([Y_train[i] == 1 for i in range(n)])[0]
            assert(len(pos_ix)+len(neg_ix) == n)
            aug_pos_ix = np.random.choice(pos_ix,
                                          size=len(neg_ix),
                                          replace=True)
            X_train = np.reshape(
                np.append([X_train[i] for i in neg_ix],
                          [X_train[i] for i in aug_pos_ix]),
                (len(neg_ix) + len(aug_pos_ix), -1))
            Y_train = np.append([0 for i in neg_ix], [1 for i in aug_pos_ix])
        n = len(X_train)
        logger.info("REM/nonREM ratio current fold: {}"
                    .format(
                        np.sum([Y_train[i] == 1 for i in range(n)])
                        / float(np.sum([Y_train[i] == 0 for i in range(n)]))
                    ))
        print("Fit the estimator")
        try:   # GCN fit takes X_val and Y_val in arguments but not the others
            estimator.fit(X_train, Y_train, X_test, Y_test,
                          "_across_testsubj_{}".format(s))
        except:
            estimator.fit(X_train, Y_train)
        print("Calculating the metrics")
        pred = estimator.predict(X_test)
        roc_auc.append(roc_auc_score(Y_test,
                                     estimator.predict_proba(X_test)[:, 1]
                                     ))
        accuracyCV.append(accuracy_score(Y_test, pred))
        Y_test = np.asarray(Y_test)
        pos_acc = accuracy_score(Y_test[Y_test == 1], pred[Y_test == 1])
        neg_acc = accuracy_score(Y_test[Y_test == 0], pred[Y_test == 0])
        bal_acc.append((pos_acc+neg_acc)/2)
        conf = confusion_matrix(Y_test, pred)
        confusion.append(conf)
        true_freq = np.reshape(np.repeat(np.sum(conf, 1), 2, axis=0), (2, 2))
        conf_perc.append(np.nan_to_num(conf.astype(float)/true_freq))
        print(roc_auc[-1], accuracyCV[-1])
    metrics = pd.DataFrame()
    metrics['balanced_acc'] = bal_acc
    metrics['auc'] = roc_auc
    metrics['accuracy'] = accuracyCV
    metrics.index = subject_list
    results = pd.DataFrame()
    results['mean'] = [np.mean(roc_auc), np.mean(accuracyCV), np.mean(bal_acc)]
    results['std'] = [np.std(roc_auc), np.std(accuracyCV), np.std(bal_acc)]
    results['min'] = [np.min(roc_auc), np.min(accuracyCV), np.min(bal_acc)]
    results['max'] = [np.max(roc_auc), np.max(accuracyCV), np.max(bal_acc)]
    results.index = ['roc_auc', 'accuracy', 'balanced_accuracy']
    return(results, metrics, confusion, conf_perc)


# ---------------------- WITHIN CV ---------------------- #
def WithinOneSubjectCV(estimator, logger, subject,
                       k=4, upsample=False, mat='std'):
    """ Implements custom within subject CV.
    Note:
        Stratified Kfold on the set of observations
        from the selected subject.
        Provide one single subject for within single subject CV.
        Provide more subjects for wihtin mixed subject CV.

    Warning:
        - Estimator has to be child BaseEstimator class
        from sklearn.
        - If you use GCN estimator the type of feature matrix
        has to be standard frequency band aggregation i.e.
        type std.
        - Assumes that all the matrices are already prepared
        per subject (see Readme)

    Args:
        estimator: estimator to use
        logger: logger to print the results to.
        subject_list: list of str with the subject names
                      to include in the CV.
        k: number of fold to use
        mat: type of feature matrix.
        upsample: whether to use upsampling or not.

    Returns:
         results: dataframe containing the
                 results averaged over all folds
        metrics: dataframe containing the
                 results per fold
        confusion: a list of confusion matrix for each fold
        conf_perc: a list of percentage confusion matrix for each fold
    """
    roc_auc = []
    accuracyCV = []
    confusion = []
    conf_perc = []
    bal_acc = []
    first = True
    for s in subject:
        if first:
            X = np.load(cwd+'/matrices/{}/{}.npy'.format(mat, s))
            Y = np.load(cwd+'/matrices/y/{}.npy'.format(s))
            first = False
        else:
            X = np.concatenate(
                (X, np.load(cwd+'/matrices/{}/{}.npy'.format(mat, s))),
                axis=0
                )
            Y = np.concatenate(
                (Y, np.load(cwd+'/matrices/y/{}.npy'.format(s))),
                axis=0
                )
    Y = [1 if ((y == 1) or (y == 2)) else 0 for y in Y]
    print(np.shape(X))
    if upsample:
        print('up')
        cv_gen = UpsampleStratifiedKFold(k)
    else:
        cv_gen = StratifiedKFold(k, shuffle=True, random_state=42)
    nsubj = len(subject)
    fold = 0
    for train_index, test_index in cv_gen.split(X, Y):
        fold += 1
        X_train = [X[i] for i in train_index]
        X_test = [X[i] for i in test_index]
        Y_train = [Y[i] for i in train_index]
        Y_test = [Y[i] for i in test_index]
        # check and log REM/nonREM proportion
        logger.info("REM/nonREM ratio current train fold: {}"
                    .format(
                        np.sum([Y_train[i] == 1 for i in range(len(X_train))])
                        / float(np.sum(
                            [Y_train[i] == 0 for i in range(len(X_train))]))
                            ))
        logger.info("REM/nonREM ratio current test fold: {}"
                    .format(
                        np.sum([Y_test[i] == 1 for i in range(len(X_test))])
                        / float(np.sum(
                                [Y_test[i] == 0 for i in range(len(X_test))]
                                ))
                        ))
        try:  # GCN.fit takes X_val and Y_val in arguments but not the others
            if nsubj == 1:  # for the file names
                estimator.fit(X_train, Y_train, X_test, Y_test,
                              "_within_subj_{}_fold_{}"
                              .format(subject[0], fold))
            else:   # for the file names
                estimator.fit(X_train, Y_train, X_test, Y_test,
                              "_within_{}_subjects_fold_{}"
                              .format(nsubj, fold))
        except:
            estimator.fit(X_train, Y_train)
        print("Calculating the metrics")
        pred = estimator.predict(X_test)
        roc_auc.append(roc_auc_score(Y_test,
                                     estimator.predict_proba(X_test)[:, 1]))
        accuracyCV.append(accuracy_score(Y_test, pred))
        Y_test = np.asarray(Y_test)
        pos_acc = accuracy_score(Y_test[Y_test == 1], pred[Y_test == 1])
        neg_acc = accuracy_score(Y_test[Y_test == 0], pred[Y_test == 0])
        bal_acc.append((pos_acc+neg_acc)/2)
        conf = confusion_matrix(Y_test, pred)
        confusion.append(conf)
        true_freq = np.repeat(np.sum(conf, 1), 2, axis=0)
        true_freq = np.reshape(true_freq, (2, 2))
        conf_perc.append(np.nan_to_num(conf.astype(float)/true_freq))
        print(roc_auc[-1], accuracyCV[-1], bal_acc[-1])
    metrics = pd.DataFrame()
    metrics['balanced_acc'] = bal_acc
    metrics['auc'] = roc_auc
    metrics['accuracy'] = accuracyCV
    results = pd.DataFrame()
    results['mean'] = [np.mean(roc_auc), np.mean(accuracyCV), np.mean(bal_acc)]
    results['std'] = [np.std(roc_auc), np.std(accuracyCV), np.std(bal_acc)]
    results['min'] = [np.min(roc_auc), np.min(accuracyCV), np.min(bal_acc)]
    results['max'] = [np.max(roc_auc), np.max(accuracyCV), np.max(bal_acc)]
    results.index = ['roc_auc', 'accuracy', 'balanced_accuracy']
    return(results, metrics, confusion, conf_perc)
