import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score

from siamese_gcn.data_utils import build_onegraph_A, ToTorchDataset

""" This file defines the training loop, a single training step
and a single validation step. This is the main helper file for
the fit/predict functions in the GCN_estimator file.
"""


def training_step(gcn, data, optimizer, criterion):
    """ Defines a single training step for one batch of observations.
    Note:
        1. Computes the Ahat matrix for each frequency band,
           for each observation in the batch.
        2. Gets the outputs from the network with this input
        3. Apply the optimizer
        4. Compute the training balanced accuracy over the batch

    Args:
        gcn(GraphConvNetwork): graph convolution network object
        data(dataframe): one batch of data (outfrom ToTorchDataset)
                i.e. dataframe with 6 columns one for the labels
                and one contains the compressed array of the
                coherence matrix for each frequency band.
        optimizer: PyTorch Optimizer object to apply
        criterion: loss function to use for the optimizer.

    Returns:
        loss: the training loss
        bal: balanced accuracy over the batch
    """
    gcn.train()

    # get the inputs
    labels = data['Y']
    coh_array1 = data['f1']
    coh_array2 = data['f2']
    coh_array3 = data['f3']
    coh_array4 = data['f4']
    coh_array5 = data['f5']

    # initialize the adjacency matrices
    n, m = coh_array1.size()
    A1 = torch.zeros((n, 90, 90))
    A2 = torch.zeros((n, 90, 90))
    A3 = torch.zeros((n, 90, 90))
    A4 = torch.zeros((n, 90, 90))
    A5 = torch.zeros((n, 90, 90))

    # we don't have feature so use identity for each graph
    X = torch.eye(90).expand(n, 90, 90)
    for i in range(n):
        A1[i] = torch.tensor(build_onegraph_A(coh_array1[i]))
        A2[i] = torch.tensor(build_onegraph_A(coh_array2[i]))
        A3[i] = torch.tensor(build_onegraph_A(coh_array3[i]))
        A4[i] = torch.tensor(build_onegraph_A(coh_array4[i]))
        A5[i] = torch.tensor(build_onegraph_A(coh_array5[i]))

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = gcn(X, A1, A2, A3, A4, A5)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    _, predicted = torch.max(outputs.data, 1)
    predicted = predicted.numpy()
    labels = labels.numpy()
    pos_acc = accuracy_score(labels[labels == 1], predicted[labels == 1])
    neg_acc = accuracy_score(labels[labels == 0], predicted[labels == 0])
    bal = (pos_acc+neg_acc)/2
    return(loss.item(), bal)


def val_step(gcn, valloader, criterion, logger):
    """ Defines a single validation step for the validation set.
    Note:
        1. Computes the Ahat matrix for each frequency band
           for each observation in the batch.
        2. Gets the outputs from the network with this input
        3. Compute the validation balanced accuracy and loss over the batch

    Args:
        gcn(GraphConvNetwork): graph convolution network object
        valloader: an PyTorch iterator containing batches of the validation set
        criterion: loss function
        logger: logger to print the results to.

    Returns:
        loss: validation loss
        acc: validation accuracy
        bal_acc: validation balanced accuracy
    """
    gcn.eval()
    loss_val = 0
    y_true = np.asarray([])
    pred_val = np.asarray([])
    c = 0
    with torch.no_grad():
        for data in valloader:
            # get the inputs
            labels = data['Y']
            coh_array1 = data['f1']
            coh_array2 = data['f2']
            coh_array3 = data['f3']
            coh_array4 = data['f4']
            coh_array5 = data['f5']

            n, m = coh_array1.size()
            A1 = torch.zeros((n, 90, 90))
            A2 = torch.zeros((n, 90, 90))
            A3 = torch.zeros((n, 90, 90))
            A4 = torch.zeros((n, 90, 90))
            A5 = torch.zeros((n, 90, 90))
            X = torch.eye(90).expand(n, 90, 90)
            for i in range(n):
                A1[i] = torch.tensor(build_onegraph_A(coh_array1[i]))
                A2[i] = torch.tensor(build_onegraph_A(coh_array2[i]))
                A3[i] = torch.tensor(build_onegraph_A(coh_array3[i]))
                A4[i] = torch.tensor(build_onegraph_A(coh_array4[i]))
                A5[i] = torch.tensor(build_onegraph_A(coh_array5[i]))
            y_true = np.append(y_true, labels)
            outputs_val = gcn(X, A1, A2, A3, A4, A5)
            _, predicted = torch.max(outputs_val.data, 1)
            pred_val = np.append(pred_val, predicted.cpu().numpy())
            loss_val += criterion(outputs_val, labels).item()
            c += 1.0
        pos_acc = accuracy_score(y_true[y_true == 1], pred_val[y_true == 1])
        neg_acc = accuracy_score(y_true[y_true == 0], pred_val[y_true == 0])
        bal = (pos_acc+neg_acc) / 2
        acc = accuracy_score(y_true, pred_val)
        logger.debug('Val loss is: %.3f' % (loss_val/c))
        logger.debug('Accuracy of the network val set : %.3f%%' % (100*acc))
        logger.debug('Balanced accuracy of the network val set : %.3f%%'
                     % (100 * bal))
    return(loss_val/c, acc, bal)


def training_loop(gcn, X_train, Y_train, batch_size, lr,
                  logger, checkpoint_dir, filename="",
                  X_val=None, Y_val=None, nsteps=1000):
    """Function that runs the training loop.

    Note:
        Assumes standard frequency band aggregation
        preprocessing step for the feature matrix
    Args:
        gcn(GraphConvNetwork): graph convolution network object
        X_train([ntrain, 5*nchannels]): training feature matrix
        Y_train([ntrain]): corresponding labels
        batch_size: batch_size
        lr(float): learning rate for the optimizer
        logger(logger): logger object to print the results to.
        checkpoint_dir(str): name of the checkpoint directory for the run.
        filename(str): filename prefix to use for the plots
        X_val([nval, 5*nchannels]): validation feature matrix
        Y_val([nval]): corresponding labels
        nsteps: number of training steps to apply
    """
    train = ToTorchDataset(X_train, Y_train)
    # If you provide a validation set during training
    if X_val is not None:
        val = ToTorchDataset(X_val, Y_val)
        valloader = torch.utils.data.DataLoader(
                                            val,
                                            batch_size=batch_size,
                                            shuffle=False)

    # Creating the batches
    torch.manual_seed(42)  # for reproducibility
    trainloader = torch.utils.data.DataLoader(
                                            train,
                                            batch_size=batch_size,
                                            shuffle=True)

    # Define loss and optimizer
    n0 = np.sum([y == 0 for y in Y_train])
    n1 = np.sum([y == 1 for y in Y_train])
    n = n1 + n0
    print(n0, n1)
    print([n/n0, n/n1])
    weight = torch.tensor([n/n0, n/n1], dtype=torch.float32)
    # CrossEntropyLoss applies softmax + cross entropy
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = optim.Adam(gcn.parameters(), lr=lr, weight_decay=5e-4)

    # Lists for saving the losses
    losses = []
    train_bal_acc = []
    loss_val = []
    acc_val = []
    step_val = []
    bal_acc_val = []

    # Training loop
    train_step = 0
    while(train_step < nsteps):
        for data in trainloader:
            if(train_step < nsteps):
                current_loss, current_balacc = training_step(
                                                gcn,
                                                data,
                                                optimizer,
                                                criterion)
                train_step += 1
                losses += [current_loss]
                train_bal_acc += [current_balacc]
                # Display the training and validation loss every 5 steps
                if (train_step % 5 == 0):
                    logger.debug("Training loss for step %d is %.3f"
                                 % (train_step, current_loss))
                    logger.debug(
                        "Training balanced accuracy for step %d is %.3f%%"
                        % (train_step, 100 * current_balacc))
                    if X_val is not None:
                        loss_val_e, acc_e, bal_e = val_step(
                                                        gcn,
                                                        valloader,
                                                        criterion,
                                                        logger)
                        loss_val.append(loss_val_e)
                        step_val.append(train_step)
                        acc_val.append(acc_e)
                        bal_acc_val.append(bal_e)

    # Save the training and validation loss and balanced accuracy plots
    if X_val is not None:
        plt.clf()
        plt.plot(losses)
        plt.plot(step_val, loss_val)
        plt.savefig(checkpoint_dir+"{}_loss.png".format(filename))
        plt.clf()
        plt.plot(train_bal_acc)
        plt.plot(step_val, bal_acc_val)
        plt.savefig(checkpoint_dir+"{}_bal_acc.png".format(filename))
