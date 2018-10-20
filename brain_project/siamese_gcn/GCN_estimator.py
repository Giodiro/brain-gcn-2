import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin

from siamese_gcn.model import GraphClassificationNet
from siamese_gcn.train_utils import training_loop
from siamese_gcn.data_utils import ToTorchDataset, \
                                   data_to_matrices


""" This file defines a wrapper class for the GCN.
This is necessary in order to use this network just as
it was any sklearn estimator in the (custom)
cross validation.
"""


class GCN_estimator_wrapper(BaseEstimator, ClassifierMixin):
    """ Wrapper for the Graph Convolutional network.
    """
    def __init__(self, checkpoint_dir, logger,
                 h1=None, h2=None, out=None,
                 in_feat=90, batch_size=64,
                 lr=0.001, nsteps=1000,
                 reset=False):
        """ Init the model from GraphClassificationNet object.
        Args:
            checkpoint_dir(str): name of the checkpoint
                                directory for the run.
            logger(logger): logger object to print the results to.
            h1: dimension of the first hidden layer
            h2: dimension of the second hidden layer
            out: dimension of the node features
            in_feat: dimension of the input features
                    i.e. number of nodes in the graph
            batch_size: batch_size
            lr(float): learning rate for the optimizer
            nsteps: number of training steps to apply
            reset(bool):  whether to reset the network
                          each time fit is called.
                          Set to true in cross-validation setting.
        """
        self.gcn = GraphClassificationNet(90, h1, h2, out)
        self.batch_size = batch_size
        self.lr = lr
        self.nsteps = nsteps
        self.checkpoint_dir = checkpoint_dir
        self.logger = logger
        self.h1 = h1
        self.h2 = h2
        self.out = out
        self.reset = reset
        logger.info("Success init of GCN params {}-{}-{}"
                    .format(self.h1, self.h2, self.out))
        logger.info("Training parameters {} steps and {} learning rate"
                    .format(self.nsteps, self.lr))

    def fit(self, X_train, Y_train, X_val=None, Y_val=None, filename=""):
        """ Train method for the network.
        This is a wrapper for the training loop.
        Note:
            Assumes standard frequency band aggregation
            preprocessing step for the feature matrix
        Args:
            X_train([ntrain, 5*nchannels]): training features matrix
            Y_train([ntrain]): corresponding labels
            X_val([nval, 5*nchannels]): validation feature matrix
            Y_val([nval]): corresponding labels
        """
        if self.reset:
            self.gcn = GraphClassificationNet(90, self.h1, self.h2, self.out)
        training_loop(self.gcn, X_train, Y_train,
                      self.batch_size, self.lr,
                      self.logger, self.checkpoint_dir, filename,
                      X_val, Y_val, nsteps=self.nsteps)

    def predict(self, X_test):
        """ Method to predict the class labels.
        Note:
            Assumes standard frequency band aggregation
            preprocessing step for the feature matrix
        Args:
            X_test([X_test, 5*nchannels]): test features matrix
        Returns:
            labels: array of 0,1 class labels.
         """
        self.gcn.eval()
        test = ToTorchDataset(np.asarray(X_test))
        testloader = torch.utils.data.DataLoader(test,
                                                 batch_size=self.batch_size,
                                                 shuffle=False,
                                                 num_workers=4)
        y_pred = []
        with torch.no_grad():
            for data in testloader:
                X, A1, A2, A3, A4, A5 = data_to_matrices(data)
                outputs = self.gcn(X, A1, A2, A3, A4, A5)
                _, predicted = torch.max(outputs.data, 1)
                y_pred.append(predicted.cpu().numpy())
        labels = np.asarray(np.concatenate(y_pred))
        return(labels)

    def predict_proba(self, X_test):
        """ Method to predict the class probabilities.
        Note:
            Assumes standard frequency band aggregation
            preprocessing step for the feature matrix
        Args:
            X_test([X_test, 5*nchannels]): test features matrix
        Returns:
            out_proba: array of probabilities of class 1.
        """
        self.gcn.eval()
        test = ToTorchDataset(X_test, None)
        testloader = torch.utils.data.DataLoader(test,
                                                 batch_size=self.batch_size,
                                                 shuffle=False,
                                                 num_workers=4)
        proba = []
        with torch.no_grad():
            for data in testloader:
                X, A1, A2, A3, A4, A5 = data_to_matrices(data)
                outputs = self.gcn(X, A1, A2, A3, A4, A5)
                proba.append(outputs.data.cpu().numpy())
        out_proba = np.asarray(np.concatenate(proba, 0))
        return(out_proba)
