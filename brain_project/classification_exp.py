import os
import numpy as np
import pandas as pd
import logging
import time
import argparse

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold, SelectPercentile
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

from siamese_gcn.GCN_estimator import GCN_estimator_wrapper
from CV_utils import WithinOneSubjectCV, AcrossSubjectCV

"""
Main file for the running experiments mentioned in the report.
"""

# ---------------------- PARAMS SETUP ---------------------- #
parser = argparse.ArgumentParser()

parser.add_argument("-est",
                    "--estimatorlist",
                    nargs='*',
                    help="list of estimator among uniform, \
                          constant, gcn, pcasvm, rf")

parser.add_argument("-s", "--nsteps",
                    help="number of steps for gcn training", type=int)
parser.add_argument("-h1", "--h1",
                    help="dimension hidden layer 1", type=int)
parser.add_argument("-h2", "--h2",
                    help="dimension hidden layer 2", type=int)
parser.add_argument("-out", "--out",
                    help="dimension features nodes", type=int)
parser.add_argument("-up", "--upsample",
                    help="if you want upsampling in the CVs", type=bool)
parser.add_argument("-j", "--njobs",
                    help="number of jobs for sklearn", type=int)
parser.add_argument("-t", "--type",
                    help="choose the preprocessing, one or std aggregation")
args = parser.parse_args()

mat = args.type if args.type else 'std'
upsample = args.upsample if args.upsample else False
njobs = args.njobs if args.njobs else 3
nsteps = args.nsteps if args.nsteps else 1500
h1 = args.h1 if args.h1 else 128
h2 = args.h2 if args.h2 else 32
out = args.out if args.out else 16

try:  # for local run
    os.chdir("/Users/melaniebernhardt/Documents/brain_project/")
    cwd = os.getcwd()
except:   # for cluster run
    cwd = os.getcwd()

# creating the directory for the run
time.time()
t = time.strftime('%d%b%y_%H%M%S')
checkpoint_dir = cwd + '/runs/' + t + '/'
os.makedirs(checkpoint_dir)
print('Saving to ' + checkpoint_dir)


# ---------------------- LOGGER SETUP ---------------------- #
# create logger
global logger
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
logger = logging.getLogger('my_log')
# create console handler and set level to debug
ch = logging.StreamHandler()
# create formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)
log_filename = checkpoint_dir + '{}_long'.format(mat) + '.log'
log_filename_short = checkpoint_dir + '{}_short'.format(mat) + '.log'
file_handler = logging.FileHandler(log_filename)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
file_handler2 = logging.FileHandler(log_filename_short)
file_handler2.setFormatter(formatter)
file_handler2.setLevel(logging.INFO)
logger.addHandler(file_handler2)

if upsample:
    logger.warning("Using upsampling")
logger.info("Using {} feature matrix".format(mat))
# ---------------------- CLASSIFIERS DEFINITION ---------------------- #

args_to_est = {'uniform': DummyClassifier(strategy='uniform'),
               'constant': DummyClassifier(constant=0, strategy='constant'),
               'pcasvm': Pipeline([
                        ('var', VarianceThreshold(threshold=0)),
                        ('pca', PCA(n_components=500)),
                        ('std', StandardScaler()),
                        ('svm', SVC(kernel='linear', probability=True))
                    ]),
               'rf': Pipeline([
                        ('var', VarianceThreshold(threshold=0)),
                        ('std', StandardScaler()),
                        ('PerBest', SelectPercentile(percentile=50)),
                        ('rf', RandomForestClassifier(
                                            n_estimators=10000,
                                            min_samples_split=30,
                                            n_jobs=4))
                        ]),
               'gcn': GCN_estimator_wrapper(checkpoint_dir, logger, h1,
                                            h2, out, nsteps=nsteps,
                                            reset=True)
               }

try:
    if args.estimatorlist:
        estimatorlist = args.estimatorlist
    else:
        estimatorlist = ['gcn', 'rf', 'pcasvm']
    print(args.estimatorlist)
    estimators = []
    for a in estimatorlist:
        estimators.append(args_to_est[a])
    names = estimatorlist
    logger.debug(estimators)
    logger.info(names)
except:
    logger.error("You provided a wrong argument for estimator list")

# ---------------------- WITHIN ONE SUBJECT CV ---------------------- #
reliable_subj = ['S04', 'S05', 'S06', 'S07', 'S08', 'S10', 'S11', 'S12']

for subject in reliable_subj:
    for i in range(len(estimators)):
        print(subject)
        results, metrics, confusion, conf_perc = WithinOneSubjectCV(
                                                    estimators[i], logger,
                                                    subject=[subject], k=4,
                                                    mat=mat,
                                                    upsample=upsample)
        logger.info("Results for subject {} for estimator {}"
                    .format(subject, names[i]))
        metrics.to_csv(checkpoint_dir+"/within_{}_{}".format(subject, names[i]))
        logger.debug("Results by fold: \n" + metrics.to_string())
        for k in range(len(confusion)):
            logger.debug("Confusion matrices fold {} is: \n"
                         .format(k) + pd.DataFrame(confusion[k]).to_string())
        logger.info("Mean results for {} estimator within".format(names[i])
                    + "subject CV for subject {} are: \n"
                      .format(subject)+results.to_string())
        logger.info("Mean of confusion matrices from within"
                    + "subject CV is: \n {} \n"
                    .format(pd.DataFrame(np.mean(confusion, 0)).to_string()))
        logger.info("Mean percentage confusion matrix"
                    + "from within subject CV \n: {}"
                      .format(pd.DataFrame(np.mean(conf_perc, 0))))
        logger.info("Std of percentage confusion matrices"
                    + "from within subject CV is: \n {} \n"
                      .format(pd.DataFrame(np.std(conf_perc, 0)).to_string()))

# ---------------------- WITHIN 4 MIXED SUBJECT CV ---------------------- #
for i in range(len(estimators)):
    logger.info("Result for within subjects (mixed) CV for {} estimator"
                .format(names[i]))
    results, metrics, confusion, conf_perc = WithinOneSubjectCV(
                                             estimators[i], logger,
                                             reliable_subj, k=5,
                                             mat=mat, upsample=upsample)
    logger.debug("Results per fold: \n"+metrics.to_string())
    metrics.to_csv(checkpoint_dir + "/within_mixed_{}".format(names[i]))
    for k in range(len(confusion)):
        logger.debug("Confusion matrices fold {} is: \n"
                     .format(k) + pd.DataFrame(confusion[k]).to_string())
    logger.info("Mean results for {} estimator within mixed subject CV are: \n"
                .format(names[i])+results.to_string())
    logger.info("Mean of confusion matrices from within subject CV are: \n {} \n"
                .format(pd.DataFrame(np.mean(confusion, 0)).to_string()))
    logger.info("Mean percentage confusion matrix from within subject CV \n: {}"
                .format(pd.DataFrame(np.mean(conf_perc, 0))))
    logger.info("Std of percentage confusion matrices from within subject CV is: \n {} \n"
                .format(pd.DataFrame(np.std(conf_perc, 0)).to_string()))


# ---------------------- ACROSS 4 SUBJECT CV ---------------------- #
for i in range(len(estimators)):
    logger.info("Results for across subjects CV for {} estimator"
                .format(names[i]))
    results, metrics, confusion, conf_perc = AcrossSubjectCV(estimators[i],
                                                             logger,
                                                             reliable_subj,
                                                             mat,
                                                             upsample=upsample)
    logger.info("Results per fold: \n"+metrics.to_string())
    metrics.to_csv(checkpoint_dir + "/across_mixed_{}".format(names[i]))
    for k in range(len(confusion)):
        logger.info("Confusion matrices per folds are: \n"
                    + pd.DataFrame(confusion[k]).to_string())
    logger.info("Mean results for {} estimator with across subject CV are: \n"
                .format(names[i])+results.to_string())
    logger.info("Mean of confusion matrices from across subject CV are: \n {} \n"
                .format(pd.DataFrame(np.mean(confusion, 0)).to_string()))
    logger.info("Mean percentage confusion matrix from across subject CV \n: {}"
                .format(pd.DataFrame(np.mean(conf_perc, 0))))
    logger.info("Std of percentage confusion matrices from across subject CV is: \n {} \n"
                .format(pd.DataFrame(np.std(conf_perc, 0)).to_string()))
