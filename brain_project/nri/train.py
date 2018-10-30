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
from dataset import EEGDataset2
from data_utils import split_within_subj
from util import time_str, mkdir_p, kl_categorical, gumbel_softmax, safe_time_str, encode_onehot
from sparse_util import to_sparse, block_diag_from_ivs_torch
from models import MLPEncoder, MLPDecoder


"""  Parameters  """

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Unused
orig_data_folder = "/nfs/nas12.ethz.ch/fs1201/infk_jbuhmann_project_leonhard/cardioml/"
# Where to fetch the data from
data_folder = "/local/home/gmeanti/cardioml/dataset2/subsample5_size250_batch32"
# Where to store tensorboard logs
log_path = "gen_data/logs/"
# Subject list is used to restrict loaded data to just the listed subjects.
subject_list = ["S01", "S02"]
# This must be implemented in the dataset class (TODO)
normalization = "none"
# The number of nodes in each graph.
num_atoms = 423
# The number of time-steps per sample (this depends on the preprocessing).
num_timesteps = 250

# Temperature of the gumbel-softmax approximation
temp = 0.2
# Whether to use the hard one-hot version (TODO: need to check if it actually works).
hard = True

# Batch size
batch_size = 2
# Learning rate
lr = 0.001
# rate of exponential decay for the learning rate (applied each epoch)
lr_decay = 0.7
# Maximum number of epochs to run for
n_epochs = 1000
plot_interval = 2

encoder_hidden = 128
prior = np.array([0.92, 0.02, 0.02, 0.02, 0.02])
n_edge_types = len(prior)
dropout = 0.1
factor = False

n_classes = 6


model_name = (f"NRIClassif{safe_time_str()}_enc{encoder_hidden}_"
              f"dout{dropout}_factor{factor}")


""" Data Loading """

with open(os.path.join(data_folder, "subj_data.json"), "r") as fh:
    subj_data = json.load(fh)

tr_indices, val_indices = split_within_subj(subject_list, subj_data)
print(f"{time_str()} Subject data contains {len(subj_data)} samples coming "
      f"from subjects {set(v['subj'] for v in subj_data.values())}.\n"
      f"{len(tr_indices)} chosen for training and {len(val_indices)} "
      f"for testing.")

num_workers = 3
tr_dataset = EEGDataset2(data_folder, tr_indices, subj_data, normalization)
val_dataset = EEGDataset2(data_folder, val_indices, subj_data, normalization)

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
off_diag = np.ones([num_atoms, num_atoms]) - np.eye(num_atoms)
adj_tensor = to_sparse(torch.tensor(off_diag.astype(np.float32)).to(device))

# Encoder
encoder = MLPEncoder(n_in=num_timesteps,
                     n_hid=encoder_hidden,
                     n_out=n_edge_types,
                     do_prob=dropout,
                     factor=factor)
encoder.to(device)

# Prior
assert sum(prior) == 1.0, "Edge prior doesn't sum to 1"
log_prior = torch.tensor(np.log(prior)).float().to(device)
log_prior = log_prior.unsqueeze(0).unsqueeze(0)

# Decoder
decoder = MLPDecoder(n_in=num_timesteps,
                     n_edge_types=n_edge_types,
                     msg_hid=128,
                     msg_out=16,
                     n_hid=128,
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

def train(epoch):
    t = time.time()

    losses_kl = []; losses_rec = []

    encoder.train()
    decoder.train()
    scheduler.step()
    for batch_idx, inputs in enumerate(tr_loader):
        X = inputs["X"].to(device)
        Y = inputs["Y"].to(device)

        optimizer.zero_grad()

        logits = encoder(X, adj_tensor)
        edges = gumbel_softmax(logits, tau=temp, hard=hard)
        prob = F.softmax(logits, dim=-1)

        edges_sparse = []
        for et in range(edges.size(2)):
            values = edges[:,:,et] # B x E
            ivs_list = []
            for b in range(edges.size(0)):
                nnz = values[b].nonzero()[0] # nnz
                v = values[b][nnz]
                i = adj_tensor._indices()[:,nnz]
                s = adj_tensor.size()
                ivs_list.append((i, v, s))

            edges_sparse.append(block_diag_from_ivs_torch(ivs_list))

        output = decoder(X, edges_sparse)
        # ... Loss calculation wrt target ...
        loss_kl = kl_categorical(prob, log_prior, num_atoms)

        # Our reconstruction loss is a bit weird, not sure what a
        # statistician would say!
        loss_rec = F.cross_entropy(output, Y, reduction="elementwise_mean")
        loss = loss_kl + loss_rec

        loss.backward()
        optimizer.step()

        losses_kl.append(loss_kl.data.cpu().numpy())
        losses_rec.append(loss_rec.data.cpu().numpy())

    loss_kl = np.mean(losses_kl)
    loss_rec = np.mean(losses_rec)

    return loss_kl, loss_rec

def validate(epoch, keep_data=False):
    t = time.time()

    losses_kl = []; losses_rec = []
    if keep_data:
        data = {"edges": [], "target": [], "preds": []}

    encoder.eval()
    decoder.eval()
    for batch_idx, inputs in enumerate(val_loader):
        X = inputs["X"].to(device)
        Y = inputs["Y"].to(device)

        logits = encoder(X, adj_tensor) # batch x n_edges x n_edge_types
        edges = gumbel_softmax(logits, tau=temp, hard=hard)
        prob = F.softmax(logits, dim=-1)

        edges_sparse = []
        indices = adj._indices()
        for et in range(edges.size(1)):
            values = edges[:,:,et] # B x E
            nnz = values.nnz()
            ivs_list = []
            for b in range(edges.size(0)):
                i = values.nnz().t() # 2 x Nnz
                v = values[i.t()]
                s = adj_tensor.size()
                ivs_list.append(i, v, s)

            edges_sparse.append(cblock_diag_from_ivs_torch(ivs_list))

        output = decoder(X, edges_sparse)

        # ... Loss calculation wrt target ...
        loss_kl = kl_categorical(prob, log_prior, num_atoms)

        # Our reconstruction loss is a bit weird, not sure what a
        # statistician would say!
        loss_rec = F.cross_entropy(output, Y, reduction="elementwise_mean")

        losses_kl.append(loss_kl.data.cpu().numpy())
        losses_rec.append(loss_rec.data.cpu().numpy())

        if keep_data:
            data["edges"].append(edges)
            data["target"].append(inputs["Y"].cpu().numpy())
            data["preds"].append(output.data.cpu().numpy())

    loss_kl = np.mean(losses_kl)
    loss_rec = np.mean(losses_rec)

    if keep_data:
        data["target"] = np.concatenate(data["target"])
        data["preds"] = np.concatenate(data["preds"])
        return loss_kl, loss_rec, data

    return loss_kl, loss_rec

def training_summaries(data, epoch, summary_writer):
    import sklearn.metrics as metrics

    targets = data["target"]
    preds = np.argmax(data["preds"], axis=1)

    # Accuracy
    accuracy = metrics.accuracy_score(targets, preds, normalize=True)
    print("Acc: ", accuracy)
    summary_writer.add_scalar("accuracy", accuracy, epoch)


""" Training Loop """

summary_path = os.path.join(log_path, model_name)
summary_writer = SummaryWriter(log_dir=summary_path)
print(f"{time_str()} Writing summaries to {summary_path} every {plot_interval} epochs.")

for epoch in range(n_epochs):
    et = time.time()

    keep_data = (epoch > 0) and (epoch % plot_interval == 0)
    tr_loss_kl, tr_loss_rec = train(epoch)
    out_val = validate(epoch, keep_data=keep_data)

    if keep_data:
        val_loss_kl, val_loss_rec, data_val = out_val
        st = time.time()
        training_summaries(data,
                           epoch,
                           summary_writer)
        print(f"{time_str()} Wrote summary data to tensorboard in {time.time() - st:.2f}s.")
    else:
        val_loss_kl, val_loss_rec = out_val

    curr_lr = scheduler.get_lr()[0]

    print(f"{time_str()} Epoch {epoch} done in {time.time() - et:.2f}s - "
          f"lrate {curr_lr} - "
          f"KL tr {tr_loss_kl:.3f}, val {val_loss_kl:.3f} - "
          f"rec tr {tr_loss_rec:.3f}, val {val_loss_rec:.3f}.")

print(f"{time_str()} Finished training. Exiting.")

