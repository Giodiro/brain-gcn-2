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
from models import MLPEncoder, MLPDecoder, FastEncoder, VAEWithClasses, VAEWithClassesReconstruction, TimeseriesClassifier, MLPReconstructionDecoder

from sacred import Experiment, Ingredient
from sacred.observers import FileStorageObserver


""" Data Loading """

syndata = Ingredient("synthetic_data")

@syndata.config
def cfg():
    """ Data Generation Parameters """
    # Number of class labels
    num_clusters = 3
    # Number of timesteps to be generated for each cluster
    total_timesteps = 40000
    # The number of nodes in each graph.
    num_atoms = 5
    # The number of time-steps per sample.
    num_timesteps = 100
    # The probability of an edge
    edge_prob = 0.2
    # Lag strength (multiplier for 1-lag)
    lag_strength = 0.5
    # Some more parameters for the precision matrix
    low_edge_prob = 0.5
    high_edge_prob = 0.9

    # train/test split
    test_size = 0.2

    batch_size = 32

    """ Data Preprocessing """
    # This must be implemented in the dataset class
    normalization = "none"

@syndata.capture
def generate_data(num_clusters, total_timesteps, num_timesteps, num_atoms, edge_prob, test_size, normalization):
    # Generate the data
    print(f"{time_str()} Generating synthetic data.")
    samples, labels, precisions = gen_synthetic_tseries(
        num_clusters=num_clusters,
        num_tsteps=total_timesteps,
        sample_size=num_timesteps,
        num_nodes=num_atoms,
        edge_prob=edge_prob)

    # Split the data between train / test
    splitting = model_selection.train_test_split(
        samples, labels, test_size=test_size, stratify=labels)
    x_train, x_test, y_train, y_test = splitting

    print(f"{time_str()} After splitting we have "
          f"{len(x_train)} samples for training and {len(x_test)} "
          f"for testing.")

    tr_dataset = SyntheticDataset(x_train, y_train, normalization=normalization)
    val_dataset = SyntheticDataset(x_test, y_test, normalization="val")
    val_dataset.normalization = tr_dataset.normalization
    val_dataset.scaler = tr_dataset.scaler

    return tr_dataset, val_dataset, precisions
    # return samples, labels, precisions

@syndata.capture
def get_dataloaders(tr_dataset, val_dataset, batch_size, num_workers=0):
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

    return tr_loader, val_loader

@syndata.capture
def get_data():
    tr_dset, val_dset, precisions = generate_data()
    tr_loader, val_loader = get_dataloaders(tr_dset, val_dset)
    return tr_loader, val_loader, precisions


traininit = Ingredient("train_initializer")

@traininit.config
def cfg():
    # Learning rate
    lr = 0.0003
    # rate of exponential decay for the learning rate (applied each epoch)
    lr_decay = 0.99


@traininit.capture
def initialize_training(model, lr, lr_decay):
    ## Training Utilities
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = optim.Adam(parameters, lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=lr_decay)

    tot_params = sum([np.prod(p.size()) for p in parameters])
    print(f"{time_str()} Initialized model with {tot_params} parameters:")

    return optimizer, scheduler


ex = Experiment("test_experiment", ingredients=[syndata, traininit])
ex.observers.append(FileStorageObserver.create('gen_data/sacred'))



"""  Parameters  """


@ex.config
def cfg(synthetic_data):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    """ File Paths """
    # Where to store tensorboard logs
    log_path = "gen_data/logs/"

    """ Training Parameters """
    # Batch size
    # Maximum number of epochs to run for
    n_epochs = 1000
    # Number of epochs in between saving to tensorboard
    plot_interval = 5

    """ Model Parameters """
    # Temperature of the gumbel-softmax approximation
    temp = 0.5
    # Whether to use the hard one-hot version
    hard = True

    encoder_type = "FastEncoder" # MLPEncoder

    if encoder_type == "FastEncoder":
        encoder_hidden = [16, 32, 32]
    elif encoder_type == "MLPEncoder":
        encoder_hidden = 64
    # Here we choose the prior based on edge_prob
    prior = np.array([0.8, 0.2])
    n_edge_types = len(prior)
    #
    encoder_dropout = 0.1
    encoder_factor = False
    enc_dist_type = "svm"

    """ Extract useful things from synthetic_data """
    num_atoms = synthetic_data["num_atoms"]
    num_timesteps = synthetic_data["num_timesteps"]
    num_clusters = synthetic_data["num_clusters"]
    batch_size = synthetic_data["batch_size"]


@ex.named_config
def class_decoder():
    decoder_type = "class"
    decoder_hidden_gnn = 128
    decoder_output_gnn = 16
    decoder_hidden_mlp = 64
    decoder_dropout = 0.1


@ex.named_config
def reconstruction_decoder():
    decoder_type = "reconstruction"
    ## MLP Reconstruction
    decoder_hidden_gnn = [32, 64, 64]
    decoder_hidden_mlp = 16
    decoder_pred_steps = 5
    decoder_use_graph = True
    decoder_dropout = 0.1

    ## Classifier
    classifier_hidden_dims = [64, 32, 32]
    classifier_dropout = 0.1



@ex.capture
def get_encoder(
    encoder_type, num_timesteps, encoder_hidden, encoder_dropout,
    n_edge_types, enc_dist_type, encoder_factor):
    if encoder_type == "FastEncoder":
        return FastEncoder(n_in=num_timesteps,
                           n_hid=encoder_hidden,
                           n_out=n_edge_types,
                           do_prob=encoder_dropout,
                           dist_type=enc_dist_type)
    elif encoder_type == "MLPEncoder":
        return MLPEncoder(n_in=num_timesteps,
                          n_hid=encoder_hidden,
                          n_out=n_edge_types,
                          do_prob=encoder_dropout,
                          factor=encoder_factor)
    else:
        raise ValueError(f"Encoder type {encoder_type} not recognized.")


@ex.capture
def get_decoder(
    decoder_type, num_atoms, num_timesteps, n_edge_types, num_clusters, decoder_dropout,
    decoder_hidden_gnn, decoder_hidden_mlp, decoder_output_gnn=None, decoder_pred_steps=None,
    decoder_use_graph=None, classifier_dropout=None):

    if decoder_type == "reconstruction":
        decoder = MLPReconstructionDecoder(
            n_atoms=num_atoms,
            n_in_node=1,
            gnn_hid_list=decoder_hidden_gnn,
            mlp_hid=decoder_hidden_mlp,
            pred_steps=decoder_pred_steps,
            use_graph=decoder_use_graph,
            dropout_prob=decoder_dropout)

        classifier = TimeseriesClassifier(
            n_in_node=num_timesteps,
            mlp_hid_list=classifier_hidden_dims,
            n_classes=num_clusters,
            dropout_prob=classifier_dropout)
        return decoder, classifier

    elif decoder_type == "class":
        decoder = MLPDecoder(
           n_in=num_timesteps,
           n_edge_types=n_edge_types,
           n_atoms=num_atoms,
           msg_hid=decoder_hidden_gnn,
           msg_out=decoder_output_gnn,
           n_hid=decoder_hidden_mlp,
           n_classes=num_clusters,
           dropout_prob=decoder_dropout)
        return decoder
    else:
        raise ValueError(f"Decoder type {decoder_type} not recognized.")


@ex.capture
def get_model(encoder_type, decoder_type, num_atoms, temp, hard, prior, device):
    if decoder_type == "reconstruction":
        encoder = get_encoder()
        decoder, classifier = get_decoder()
        model = VAEWithClassesReconstruction(
            encoder=encoder,
            decoder=decoder,
            classifier=classifier,
            temp=temp,
            hard=hard,
            prior=prior)
    elif decoder_type == "class":
        encoder = get_encoder()
        decoder = get_decoder()
        model = VAEWithClasses(
           encoder=encoder,
           decoder=decoder,
           temp=temp,
           hard=hard,
           prior=prior)
    else:
        raise ValueError(f"Decoder type {decoder_type} not recognized.")

    model.to(device)

    # Generate the fully-connected graph which is initially used for interactions.
    # We use the upper triangular part only since we're interested in undirected
    # graphs.
    triu_mat = np.triu(np.ones([num_atoms, num_atoms]), k=1)
    adj_tensor = to_sparse(torch.tensor(triu_mat.astype(np.float32)).to(device))

    return model, adj_tensor


@ex.capture
def get_model_name(_config):
    model_name = (f"NRIClassif{safe_time_str()}_"
                  f"enc{list_to_safe_str(_config['encoder_hidden'])}_"
                  f"dout{_config['encoder_dropout']}_factor{_config['encoder_factor']}_"
                  f"hard{_config['hard']}_temp{_config['temp']}_"
                  f"enctype{_config['encoder_type']}_dectype{_config['decoder_type']}_")
    return model_name

@ex.capture
def run_epoch(epoch, model, optimizer, scheduler, data_loader, adj_tensor, device, _run, keep_data=False, validate=False):
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
    losses_bookkeeping = {}
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
            data_dict["edges"].append(model.edges.data.cpu().numpy())
            data_dict["edge_probs"].append(model.prob.data.cpu().numpy())
            data_dict["target"].append(inputs["Y"].data.cpu().numpy())
            data_dict["preds"].append(model.output.data.cpu().numpy())

        model.delete_saved()

    for k, v in losses_bookkeeping.items():
        data_dict["loss_" + k] = np.mean(v)
        if validate:
            _run.log_scalar(f"validation.loss_{k}", np.mean(v), epoch)
        else:
            _run.log_scalar(f"training.loss_{k}", np.mean(v), epoch)

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
            out_str += f"{k} {suffix} {v:.4f} "
    return out_str


@ex.capture
def training_summaries(data_dict, epoch, summary_writer, num_atoms, suffix="val"):
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
    for k, v in data_dict.items():
        if k.startswith("loss_"):
            summary_writer.add_scalar(f"{k[5:]}/{suffix}", v, epoch)

    """ Confusion matrix """
    fig = plot_confusion_matrix(targets, preds, dataset.CLASS_NAMES)
    summary_writer.add_figure(f"conf_mat/{suffix}", fig, epoch, close=True)




@ex.automain
def main(log_path, _run, _config):
    model_name = get_model_name()
    info = _run.info
    summary_path = os.path.join(log_path, model_name)
    summary_writer = SummaryWriter(log_dir=summary_path)
    info["tensorflow"] = {"logdirs": ["summary_path"]}

    print(f"{time_str()} Writing summaries to {summary_path} every {_config['plot_interval']} epochs.")

    model, adj_tensor = get_model()
    tr_loader, val_loader, precisions = get_data()
    optimizer, scheduler = initialize_training(model)

    best_val_loss = None

    for epoch in range(_config["n_epochs"]):
        et = time.time()

        # Keep data is used as a flag specifying whether to write data
        # to tensorboard or not. It depends on the `plot_interval` parameter.
        keep_data = (epoch > 0) and (epoch % _config["plot_interval"] == 0)
        tr_data = run_epoch(
            epoch,
            model,
            optimizer,
            scheduler,
            tr_loader,
            adj_tensor,
            keep_data=keep_data,
            validate=False)
        val_data = run_epoch(
            epoch,
            model,
            optimizer,
            scheduler,
            val_loader,
            adj_tensor,
            keep_data=keep_data,
            validate=True)
        epoch_elapsed = time.time() - et
        if keep_data:
            st = time.time()
            training_summaries(
                tr_data,
                epoch,
                summary_writer,
                suffix="tr")
            training_summaries(
                val_data,
                epoch,
                summary_writer,
                suffix="val")
            print(f"{time_str()} Wrote summary data to tensorboard in {time.time() - st:.2f}s.")

        curr_lr = scheduler.get_lr()[0]

        print(f"{time_str()} Epoch {epoch:4d} done in {epoch_elapsed:.2f}s - "
              f"lrate {curr_lr:.5f} - {print_losses(tr_data, 'train')} - "
              f"{print_losses(val_data, 'val')}")

        val_loss = sum([v for k, v in val_data.items() if k.startswith("loss_")])
        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss

    print(f"{time_str()} Finished training. Lowest validation loss {best_val_loss:.5f}.")
    return best_val_loss

