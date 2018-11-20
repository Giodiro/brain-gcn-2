import os
import numpy as np
import json
import time

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn import model_selection
import networkx as nx

from tensorboardX import SummaryWriter
import dataset
from dataset import EEGDataset2, SyntheticDataset
from synthetic_data import gen_synthetic_tseries
from util import (time_str, mkdir_p, kl_categorical, safe_time_str, encode_onehot, plot_confusion_matrix, list_to_safe_str)
from sparse_util import to_sparse, block_diag_from_ivs_torch
from models import MLPEncoder, MLPDecoder, FastEncoder, VAEWithClasses, VAEWithClassesReconstruction


"""  Parameters  """

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Number of class labels
num_clusters = 3
# Number of timesteps to be generated for each cluster
total_timesteps = 40000

# Where to store tensorboard logs
log_path = "gen_data/logs/"
# This must be implemented in the dataset class
normalization = "none"
# The number of nodes in each graph.
num_atoms = 5
# The number of time-steps per sample.
num_timesteps = 100

# Temperature of the gumbel-softmax approximation
temp = 0.5
# Whether to use the hard one-hot version
hard = True

# Batch size
batch_size = 32
# Learning rate
lr = 0.0003
# rate of exponential decay for the learning rate (applied each epoch)
lr_decay = 0.99
# Maximum number of epochs to run for
n_epochs = 1000
plot_interval = 5

encoder_hidden = [16, 32, 32]
# Here we choose the prior based on edge_prob
prior = np.array([0.8, 0.2])
n_edge_types = len(prior)
dropout = 0.1
factor = False
enc_dist_type = "svm"

decoder_hidden1 = 128
decoder_hidden2 = 64
decoder_out = 16


model_name = (f"NRIClassif{safe_time_str()}_"
              f"enc{list_to_safe_str(encoder_hidden)}_"
              f"dout{dropout}_factor{factor}_"
              f"hard{hard}_temp{temp}_"
              f"dist_type{enc_dist_type}_"
              f"dec{list_to_safe_str([decoder_hidden1, decoder_hidden2, decoder_out])}")


""" Data Loading """

# Generate the data
print(f"{time_str()} Generating synthetic data.")
samples, labels = gen_synthetic_tseries(num_clusters=num_clusters,
                                        num_tsteps=total_timesteps,
                                        sample_size=num_timesteps,
                                        num_nodes=num_atoms,
                                        edge_prob=0.2)

# Split the data between train / test
splitting = model_selection.train_test_split(
    samples, labels, test_size=0.2, stratify=labels)
x_train, x_test, y_train, y_test = splitting

print(f"{time_str()} After splitting we have "
      f"{len(x_train)} samples for training and {len(x_test)} "
      f"for testing.")

tr_dataset = SyntheticDataset(x_train, y_train, normalization)
val_dataset = SyntheticDataset(x_test, y_test, normalization="val")
# Normalization statistics should only be computed on the training set.
val_dataset.normalization = tr_dataset.normalization
val_dataset.scaler = tr_dataset.scaler

tr_loader = data.DataLoader(tr_dataset,
                            shuffle=True,
                            batch_size=batch_size,
                            num_workers=0,
                            collate_fn=SyntheticDataset.collate,
                            pin_memory=False)

val_loader = data.DataLoader(val_dataset,
                             shuffle=True,
                             batch_size=batch_size,
                             num_workers=0,
                             collate_fn=SyntheticDataset.collate,
                             pin_memory=False)

print(f"{time_str()} Initialized data loaders with batch size {batch_size}.")


""" Initialize Models """

# Generate the fully-connected graph which is initially used for interactions.
# We use the upper triangular part only since we're interested in undirected
# graphs.
triu_mat = np.triu(np.ones([num_atoms, num_atoms]), k=1)
adj_tensor = to_sparse(torch.tensor(triu_mat.astype(np.float32)).to(device))

## Encoder
#encoder = MLPEncoder(n_in=num_timesteps,
#                     n_hid=64,
#                     n_out=n_edge_types,
#                     do_prob=dropout,
#                     factor=factor)
encoder = FastEncoder(n_in=num_timesteps,
                      n_hid=encoder_hidden,
                      n_out=n_edge_types,
                      do_prob=dropout,
                      dist_type=enc_dist_type)


## Decoder
decoder = MLPDecoder(n_in=num_timesteps,
                     n_edge_types=n_edge_types,
                     n_atoms=num_atoms,
                     msg_hid=decoder_hidden1,
                     msg_out=decoder_out,
                     n_hid=decoder_hidden2,
                     n_classes=num_clusters,
                     dropout_prob=dropout)

model = VAEWithClasses(
    encoder=encoder,
    decoder=decoder,
    temp=temp,
    hard=hard,
    prior=prior)


# model = VAEWithClassesReconstruction(
#     encoder=encoder,
#     decoder=decoder,
#     classifier=classifier,
#     temp=temp,
#     hard=hard,
#     prior=prior)

model.to(device)

## Training Utilities
parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
optimizer = optim.Adam(parameters, lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=lr_decay)

tot_params = sum([np.prod(p.size()) for p in parameters])
print(f"{time_str()} Initialized model with {tot_params} parameters:")


""" Train / Validation Functions """


def run_epoch(epoch, data_loader, keep_data=False, validate=False):
    """Run the model for an epoch.
    Args:
     - epoch : int
        The current epoch
     - data_loader
        PyTorch data loader to iterate over the batches
     - keep_data : bool (default False)
        Whether to keep batch outputs. Only necessary if we later
        want to calculate statistics and report them.
     - validate : bool (default False)
        Whether to train the model or not.
    """
    data_dict = {}
    if keep_data:
        data_dict = {"edges": [], "target": [], "preds": [], "edge_probs": []}

    if validate:
        model.eval()
    else:
        model.train()
        scheduler.step()

    for batch_idx, inputs in enumerate(data_loader):
        X = inputs["X"].to(device)
        Y = inputs["Y"].to(device)

        if not validate:
            optimizer.zero_grad()

        # Run the model & Calculate the loss
        losses_dict = model(X, adj_tensor, Y)

        if not validate:
            #kl_proportion = torch.tensor(max(np.exp(-epoch/30), 0.5)).to(loss_kl.device)
            #loss = kl_proportion * losses_dict["KL"] + (1 - kl_proportion) * losses_dict["Reconstruction"]
            # Call to the optimizer
            loss = sum(losses_dict.values())
            loss.backward()
            optimizer.step()

        # Lots of book-keeping from here
        for k, v in losses_dict.items():
            try:
                losses_bookkeeping[k].append(v.data.cpu().numpy())
            except KeyError:
                losses_bookkeeping[k] = [v.data.cpu().numpy()]

        if keep_data:
            data_dict["edges"].append(emodel.dges.data.cpu().numpy())
            data_dict["edge_probs"].append(model.prob.data.cpu().numpy())
            data_dict["target"].append(inputs["Y"].data.cpu().numpy())
            data_dict["preds"].append(model.output.data.cpu().numpy())

        model.delete_saved()

    for k, v in losses_bookkeeping.items():
        data_dict["loss_" + k] = np.mean(v)

    if keep_data:
        data_dict["edges"] = np.concatenate(data_dict["edges"])
        data_dict["edge_probs"] = np.concatenate(data_dict["edge_probs"])
        data_dict["target"] = np.concatenate(data_dict["target"])
        data_dict["preds"] = np.concatenate(data_dict["preds"])

    return data_dict


def print_losses(data_dict, suffix):
    out_str = ""
    for k, v in data_dict.items():
        if k.startswith("loss_"):
            out_str += f"{k} {suffix} {v:.4f}"
    return out_str


def training_summaries(data_dict, epoch, summary_writer, suffix="val"):
    """Write summary statistics to tensorboard.
    """
    import sklearn.metrics as metrics

    targets = data_dict["target"]
    preds = np.argmax(data_dict["preds"], axis=1)

    """ Accuracy """
    accuracy = metrics.accuracy_score(targets, preds, normalize=True)
    summary_writer.add_scalar(f"accuracy/{suffix}", accuracy, epoch)

    """ Edges (bar-chart) """
    edges = data_dict["edges"]
    edges = edges.reshape(-1, edges.shape[-1])
    edge_sums = np.mean(edges, axis=0)

    fig, ax = plt.subplots()
    ax.bar(
        np.arange(len(edge_sums)),
        edge_sums,
        align="center",
        alpha=0.7)
    ax.set_xticks(np.arange(len(edge_sums)))
    ax.set_xticklabels(["E%d" % i for i in range(len(edge_sums))])

    summary_writer.add_figure(f"edge_types/{suffix}", fig, epoch, close=True)

    """Another bar chart taking the mean of the edge probabilities
    which are calculated with softmax instead of gumbel softmax.
    """
    eprob = data_dict["edge_probs"]
    eprob = eprob.reshape(-1, eprob.shape[-1])
    eprob = np.mean(eprob, axis=0)
    fig, ax = plt.subplots()
    ax.bar(
        np.arange(len(edge_sums)),
        edge_sums,
        align="center",
        alpha=0.7)
    ax.set_xticks(np.arange(len(edge_sums)))
    ax.set_xticklabels(["E%d" % i for i in range(len(edge_sums))])

    summary_writer.add_figure(f"softmax_edge_types/{suffix}", fig, epoch, close=True)


    """ Prototypical Graphs """
    edges = data_dict["edges"]
    for target in np.unique(targets):
        tedges = edges[targets == target] # B x num_edges x num_edge_types
        tedges = np.mean(tedges, 0) # num_edges x num_edge_types
        etype = 1
        tedges = tedges[:,etype]
        # Create graph from edges
        G = nx.Graph()
        G.add_nodes_from(np.arange(num_atoms))
        k = 0
        for i in range(num_atoms):
            for j in range(i+1, num_atoms):
                G.add_edge(i, j, weight=tedges[k])
                k += 1
        fig, ax = plt.subplots()

        pos = nx.circular_layout(G)
        nx.draw_networkx(G, pos=pos, ax=ax, with_labels=True)
        nx.draw_networkx_edge_labels(G, pos=pos, ax=ax, edge_labels=nx.get_edge_attributes(G, 'weight'))

        summary_writer.add_figure(f"edge{etype}_target{target}/{suffix}", fig, epoch, close=True)


    """ Losses """
    summary_writer.add_scalar(f"KL_loss/{suffix}", data_dict["KL_loss"], epoch)
    summary_writer.add_scalar(f"cross_entropy_loss/{suffix}", data_dict["rec_loss"], epoch)

    """ Confusion matrix """
    fig = plot_confusion_matrix(targets, preds, dataset.CLASS_NAMES)
    summary_writer.add_figure(f"conf_mat/{suffix}", fig, epoch, close=True)


""" Training Loop """

summary_path = os.path.join(log_path, model_name)
summary_writer = SummaryWriter(log_dir=summary_path)
print(f"{time_str()} Writing summaries to {summary_path} every {plot_interval} epochs.")

for epoch in range(n_epochs):
    et = time.time()

    # Keep data is used as a flag specifying whether to write data
    # to tensorboard or not. It depends on the `plot_interval` parameter.
    keep_data = (epoch > 0) and (epoch % plot_interval == 0)
    tr_data = run_epoch(epoch,
                        tr_loader,
                        keep_data=keep_data,
                        validate=False)
    val_data = run_epoch(epoch,
                         val_loader,
                         keep_data=keep_data,
                         validate=True)
    epoch_elapsed = time.time() - et
    if keep_data:
        st = time.time()
        training_summaries(tr_data,
                           epoch,
                           summary_writer,
                           suffix="tr")
        training_summaries(val_data,
                           epoch,
                           summary_writer,
                           suffix="val")
        print(f"{time_str()} Wrote summary data to tensorboard in {time.time() - st:.2f}s.")

    curr_lr = scheduler.get_lr()[0]

    print(f"{time_str()} Epoch {epoch:4d} done in {epoch_elapsed:.2f}s - "
          f"lrate {curr_lr:.5f} - {print_losses(tr_data, "train")} - "
          f"{print_losses(val_data, "val")}")

print(f"{time_str()} Finished training. Exiting.")

