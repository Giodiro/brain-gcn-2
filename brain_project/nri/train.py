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

from tensorboardX import SummaryWriter
import dataset
from dataset import EEGDataset2
from data_utils import split_within_subj
from util import (time_str, mkdir_p, kl_categorical, safe_time_str, encode_onehot, plot_confusion_matrix)
from sparse_util import to_sparse, block_diag_from_ivs_torch
from models import MLPEncoder, MLPDecoder, FastEncoder


"""  Parameters  """

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Unused
orig_data_folder = "/nfs/nas12.ethz.ch/fs1201/infk_jbuhmann_project_leonhard/cardioml/"
# Where to fetch the data from
#data_folder = "/local/home/gmeanti/cardioml/dataset2/subsample10_size250_batch64"
data_folder = "/cluster/home/gmeanti/cardioml/dataset2/subsample10_size250_batch64"
# Where to store tensorboard logs
log_path = "gen_data/logs/"
# Subject list is used to restrict loaded data to just the listed subjects.
subject_list = ["S04", "S05", "S06", "S07", "S08"]
# This must be implemented in the dataset class
normalization = "standard"
# The number of nodes in each graph.
num_atoms = 423
# The number of time-steps per sample (this depends on the preprocessing).
num_timesteps = 250

# Temperature of the gumbel-softmax approximation
temp = 0.5
# Whether to use the hard one-hot version
hard = True

# Batch size
batch_size = 16
# Learning rate
lr = 0.001
# rate of exponential decay for the learning rate (applied each epoch)
lr_decay = 0.7
# Maximum number of epochs to run for
n_epochs = 1000
plot_interval = 2

encoder_hidden = [32, 64, 32]
prior = np.array([0.94, 0.02, 0.02, 0.02])
n_edge_types = len(prior)
dropout = 0.1
factor = True
enc_dist_type = "svm"

decoder_hidden1 = 32
decoder_hidden2 = 64
decoder_out = 16

n_classes = 6


model_name = (f"NRIClassif{safe_time_str()}_enc{encoder_hidden}_"
              f"dout{dropout}_factor{factor}")


""" Data Loading """

with open(os.path.join(data_folder, "subj_data.json"), "r") as fh:
    subj_data = json.load(fh)

print(f"{time_str()} Subject data contains {len(subj_data)} samples coming "
      f"from subjects {set(v['subj'] for v in subj_data.values())}.")

tr_indices, val_indices = split_within_subj(subject_list, subj_data)

print(f"{time_str()} After restricting to {subject_list} we have"
      f"{len(tr_indices)} for training and {len(val_indices)} "
      f"for testing.")

num_workers = 2
tr_dataset = EEGDataset2(data_folder, tr_indices, subj_data, normalization)
val_dataset = EEGDataset2(data_folder, val_indices, subj_data, normalization="val")
# Normalization statistics should only be computed on the training set.
val_dataset.normalization = tr_dataset.normalization
val_dataset.scaler = tr_dataset.scaler

tr_loader = data.DataLoader(tr_dataset,
                            shuffle=True,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            collate_fn=EEGDataset2.collate,
                            pin_memory=False)

val_loader = data.DataLoader(val_dataset,
                            shuffle=True,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            collate_fn=EEGDataset2.collate,
                            pin_memory=False)

print(f"{time_str()} Initialized data loaders with {num_workers} workers and "
      f"batch size of {batch_size}.")

""" Initialize Models """

# Generate off-diagonal interaction graph i.e. a fully connected graph
# without self-loops.
# Alternative: Just the upper triangular part of the graph (without the diagonal)
triu_mat = np.triu(np.ones([num_atoms, num_atoms]), k=1)
adj_tensor = to_sparse(torch.tensor(triu_mat.astype(np.float32)).to(device))

# Encoder
#encoder = MLPEncoder(n_in=num_timesteps,
#                     n_hid=encoder_hidden,
#                     n_out=n_edge_types,
#                     do_prob=dropout,
#                     factor=factor)
encoder = FastEncoder(n_in=num_timesteps,
                      n_hid=encoder_hidden,
                      n_out=n_edge_types,
                      do_prob=dropout,
                      dist_type=enc_dist_type)
encoder.to(device)

# Prior
assert sum(prior) == 1.0, "Edge prior doesn't sum to 1"
log_prior = torch.tensor(np.log(prior)).float().to(device)
log_prior = log_prior.unsqueeze(0).unsqueeze(0)

# Decoder
decoder = MLPDecoder(n_in=num_timesteps,
                     n_edge_types=n_edge_types,
                     msg_hid=decoder_hidden1,
                     msg_out=decoder_out,
                     n_hid=decoder_hidden2,
                     n_classes=n_classes,
                     dropout_prob=dropout)
decoder.to(device)

# Training
parameters = (list(filter(lambda p: p.requires_grad, encoder.parameters())) +
              list(filter(lambda p: p.requires_grad, decoder.parameters())))
optimizer = optim.Adam(parameters, lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=lr_decay)

tot_params = sum([np.prod(p.size()) for p in parameters])
print(f"{time_str()} Initialized model with {tot_params} parameters:")
#for p in parameters:
#    print(f"\tparam {p.name}: {p.size()}  (tot {np.prod(p.size())})")


""" Train / Validation Functions """

def train(epoch, keep_data=False):
    t = time.time()

    losses_kl = []; losses_rec = []
    if keep_data:
        data_dict = {"edges": [], "target": [], "preds": []}

    encoder.train()
    decoder.train()
    scheduler.step()
    for batch_idx, inputs in enumerate(tr_loader):
        X = inputs["X"].to(device)
        Y = inputs["Y"].to(device)

        optimizer.zero_grad()
        #es = time.time()
        logits = encoder(X, adj_tensor)  # B x E x Et
        #torch.cuda.synchronize()
        #ee = time.time()

        edges = F.gumbel_softmax(logits.view(-1, logits.size(2)),
                                 tau=temp, hard=hard).view(logits.size())

        prob = F.softmax(logits, dim=-1)
        loss_kl = kl_categorical(prob, log_prior, num_atoms)

        #ds = time.time()
        output = decoder(X, edges)
        # Our reconstruction loss is a bit weird, not sure what a
        # statistician would say!
        loss_rec = F.cross_entropy(output, Y, size_average=True)
        #torch.cuda.synchronize()

        #ls = time.time()
        loss = loss_kl + loss_rec
        loss.backward()
        #le = time.time()

        optimizer.step()

        #print(f"Training. Enc {ee - es:.3f} - Gumbel {ds - ee:.3f} - Dec {ls - ds:.3f} - Back {le - ls:.3f} - Tot {time.time() - es:.3f}")

        losses_kl.append(loss_kl.data.cpu().numpy())
        losses_rec.append(loss_rec.data.cpu().numpy())

        if keep_data:
            data_dict["edges"].append(edges.data.cpu().numpy())
            data_dict["target"].append(inputs["Y"].data.cpu().numpy())
            data_dict["preds"].append(output.data.cpu().numpy())

    loss_kl = np.mean(losses_kl)
    loss_rec = np.mean(losses_rec)

    if keep_data:
        data_dict["edges"] = np.concatenate(data_dict["edges"])
        data_dict["target"] = np.concatenate(data_dict["target"])
        data_dict["preds"] = np.concatenate(data_dict["preds"])
        data_dict["KL_loss"] = loss_kl
        data_dict["rec_loss"] = loss_rec
        return loss_kl, loss_rec, data_dict

    return loss_kl, loss_rec, None

def validate(epoch, keep_data=False):
    t = time.time()

    losses_kl = []; losses_rec = []
    if keep_data:
        data_dict = {"edges": [], "target": [], "preds": []}

    encoder.eval()
    decoder.eval()
    for batch_idx, inputs in enumerate(val_loader):
        X = inputs["X"].to(device)
        Y = inputs["Y"].to(device)

        logits = encoder(X, adj_tensor) # batch x n_edges x n_edge_types
        edges = F.gumbel_softmax(logits.view(-1, logits.size(2)),
                                 tau=temp, hard=hard).view(logits.size())
        prob = F.softmax(logits, dim=-1)

        output = decoder(X, edges)

        loss_kl = kl_categorical(prob, log_prior, num_atoms)
        # Our reconstruction loss is a bit weird, not sure what a
        # statistician would say!
        loss_rec = F.cross_entropy(output, Y, size_average=True)

        losses_kl.append(loss_kl.data.cpu().numpy())
        losses_rec.append(loss_rec.data.cpu().numpy())

        if keep_data:
            data_dict["edges"].append(edges.data.cpu().numpy())
            data_dict["target"].append(inputs["Y"].data.cpu().numpy())
            data_dict["preds"].append(output.data.cpu().numpy())

    loss_kl = np.mean(losses_kl)
    loss_rec = np.mean(losses_rec)

    if keep_data:
        data_dict["edges"] = np.concatenate(data_dict["edges"])
        data_dict["target"] = np.concatenate(data_dict["target"])
        data_dict["preds"] = np.concatenate(data_dict["preds"])
        data_dict["KL_loss"] = loss_kl
        data_dict["rec_loss"] = loss_rec
        return loss_kl, loss_rec, data_dict

    return loss_kl, loss_rec, None

def training_summaries(data_dict, epoch, summary_writer, suffix="val"):
    import sklearn.metrics as metrics

    targets = data_dict["target"]
    preds = np.argmax(data_dict["preds"], axis=1)

    # Accuracy
    accuracy = metrics.accuracy_score(targets, preds, normalize=True)
    summary_writer.add_scalar(f"accuracy/{suffix}", accuracy, epoch)

    # Edges (bar-chart)
    edges = data_dict["edges"].reshape(-1, edges.shape[-1])
    edge_sums = np.sum(edges, axis=0)

    fig, ax = plt.subplots()
    ax.bar(
        np.arange(len(edge_sums)),
        edge_sums,
        align="center",
        alpha=0.7)
    ax.set_xticks(np.arange(len(edge_sums)))
    ax.set_xticklabels(["E%d" % i for i in range(len(edge_sums))])

    summary_writer.add_figure(f"edge_types/{suffix}", fig, epoch, close=True)

    # Losses
    summary_writer.add_scalar(f"KL_loss/{suffix}", data_dict["KL_loss"], epoch)
    summary_writer.add_scalar(f"cross_entropy_loss/{suffix}", data_dict["rec_loss"], epoch)

    # Confusion matrix
    fig = plot_confusion_matrix(targets, preds, dataset.class_names)
    summary_writer.add_figure(f"conf_mat/{suffix}", fig, epoch, close=True)


""" Training Loop """

summary_path = os.path.join(log_path, model_name)
summary_writer = SummaryWriter(log_dir=summary_path)
print(f"{time_str()} Writing summaries to {summary_path} every {plot_interval} epochs.")

for epoch in range(n_epochs):
    et = time.time()

    keep_data = (epoch > 0) and (epoch % plot_interval == 0)
    tr_loss_kl, tr_loss_rec, tr_data = train(epoch, keep_data=keep_data)
    val_loss_kl, val_loss_rec, val_data = validate(epoch, keep_data=keep_data)

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

    print(f"{time_str()} Epoch {epoch:4d} done in {time.time() - et:.2f}s - "
          f"lrate {curr_lr:.5f} - "
          f"KL tr {tr_loss_kl:.4f}, val {val_loss_kl:.4f} - "
          f"rec tr {tr_loss_rec:.4f}, val {val_loss_rec:.4f}.")

print(f"{time_str()} Finished training. Exiting.")

