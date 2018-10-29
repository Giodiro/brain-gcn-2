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
from sparse_util import to_sparse
from models import MLPEncoder, MLPDecoder


"""  Parameters  """

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

orig_data_folder = "/nfs/nas12.ethz.ch/fs1201/infk_jbuhmann_project_leonhard/cardioml/"
data_folder = "/cluster/home/gmeanti/cardioml/dataset2/subsample5_size250_batch32"
log_path = "gen_data/logs/"
subject_list = ["S01", "S02"]
normalization = "none"
num_atoms = 423
num_timesteps = 250

temp = 0.5
hard = False

# Batch size
batch_size = 4
# Learning rate
lr = 0.001
# rate of exponential decay for the learning rate (applied each epoch)
lr_decay = 0.7
# Maximum number of epochs to run for
num_epochs = 1000
plot_interval = 5

encoder_hidden = 128
prior = np.array([0.92, 0.02, 0.02, 0.02, 0.02])
edge_types = len(prior)
encoder_dropout = 0.0
decoder_dropout = 0.0
factor = True

num_classes = 6


model_name = (f"NRIClassif{safe_time_str()}_enc{encoder_hidden}_"
              f"dout{encoder_dropout}_factor{factor}")


""" Data Loading """

with open(os.path.join(data_folder, "subj_data.json"), "r") as fh:
    subj_data = json.load(fh)

tr_indices, val_indices = split_within_subj(subject_list, subj_data)

tr_dataset = EEGDataset2(data_folder, tr_indices, subj_data, normalization)
val_dataset = EEGDataset2(data_folder, val_indices, subj_data, normalization)

tr_loader = data.DataLoader(tr_dataset,
                            shuffle=True,
                            batch_size=batch_size,
                            num_workers=3,
                            collate_fn=EEGDataset2.collate,
                            pin_memory=False)

val_loader = data.DataLoader(val_dataset,
                            shuffle=True,
                            batch_size=batch_size,
                            num_workers=3,
                            collate_fn=EEGDataset2.collate,
                            pin_memory=False)

""" Initialize Models """

# Generate off-diagonal interaction graph
off_diag = np.ones([num_atoms, num_atoms]) - np.eye(num_atoms)

rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
rel_rec = to_sparse(torch.tensor(rel_rec)).to(device)
rel_send = to_sparse(torch.tensor(rel_send)).to(device)

# Encoder
encoder = MLPEncoder(num_timesteps,
                     encoder_hidden,
                     edge_types,
                     encoder_dropout,
                     factor)
encoder.to(device)
next(encoder.parameters()).to(device)

# Prior
assert sum(prior) == 1.0, "Edge prior doesn't sum to 1"
log_prior = torch.tensor(np.log(prior)).to(device)
log_prior = log_prior.unsqueeze(0).unsqueeze(0)

# Decoder
decoder = MLPDecoder(n_in_node=num_atoms,
                     n_edge_types=edge_types,
                     msg_hid=128,
                     msg_out=16,
                     n_hid=128,
                     rnn_hid=128,
                     n_classes=num_classes,
                     dropout_prob=decoder_dropout)
next(decoder.parameters()).to(device)
encoder.to(device)

# Training
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=lr_decay)

""" Train / Validation Functions """

def train(epoch, keep_data=False):
    t = time.time()

    losses_kl = []; losses_rec = []
    if keep_data:
        data = {"edges": []}

    encoder.train()
    decoder.train()
    scheduler.step()
    for batch_idx, inputs in enumerate(tr_loader):
        X = inputs["X"].to(device)
        Y = inputs["Y"].to(device)

        optimizer.zero_grad()

        logits = encoder(X, rel_rec, rel_send)
        edges = gumbel_softmax(logits, tau=temp, hard=hard)
        prob = F.softmax(logits, dim=-1)

        output = decoder(X, edges, rel_rec, rel_send)

        # ... Loss calculation wrt target ...
        loss_kl = kl_categorical(prob, log_prior, num_atoms)

        # Our reconstruction loss is a bit weird, not sure what a
        # statistician would say!
        loss_rec = F.cross_entropy_loss(output, Y, reduction="elementwise_mean")
        loss = loss_kl + loss_rec

        loss.backward()
        optimizer.step()

        losses_kl.append(loss_kl.data.cpu().numpy())
        losses_rec.append(loss_rec.data.cpu().numpy())

        if keep_data:
            data["edges"].append(edges)

    loss_kl = np.mean(losses_kl)
    loss_rec = np.mean(losses_rec)

    print(f"{time_str()} Epoch {epoch} done in {time.time() - t:.2f}s - "
          f"loss {loss_kl:.3f} loss {loss_rec:.3f}.")

    if keep_data:
        return loss_kl, loss_rec, data

    return loss_kl, loss_rec

def validate(epoch, keep_data=False):
    t = time.time()

    losses_kl = []; losses_rec = []
    if keep_data:
        data = {"edges": []}

    encoder.eval()
    decoder.eval()
    for batch_idx, inputs in enumerate(val_loader):
        X = inputs["X"].to(device)
        Y = inputs["Y"].to(device)

        logits = encoder(X, rel_rec, rel_send)
        edges = gumbel_softmax(logits, tau=temp, hard=hard)
        prob = F.softmax(logits, dim=-1)

        output = decoder(X, edges, rel_rec, rel_send)

        # ... Loss calculation wrt target ...
        loss_kl = kl_categorical(prob, log_prior, num_atoms)

        # Our reconstruction loss is a bit weird, not sure what a
        # statistician would say!
        loss_rec = F.cross_entropy_loss(output, Y, reduction="elementwise_mean")

        losses_kl.append(loss_kl.data.cpu().numpy())
        losses_rec.append(loss_rec.data.cpu().numpy())

        if keep_data:
            data["edges"].append(edges)

    loss_kl = np.mean(losses_kl)
    loss_rec = np.mean(losses_rec)

    print(f"{time_str()} Epoch {epoch} done in {time.time() - t:.2f}s - "
          f"loss {loss_kl:.3f} loss {loss_rec:.3f}.")

    if keep_data:
        return loss_kl, loss_rec, data

    return loss_kl, loss_rec


""" Training Loop (TODO fix)"""

summary_path = os.path.join(log_path, model_name)
summary_writer = SummaryWriter(log_dir=summary_path)
print(f"{time_str()} Writing summaries to {summary_path} every {plot_interval} epochs.")

for epoch in range(num_epochs):
    keep_data = (epoch > 0) and (epoch % plot_interval == 0)
    out_tr = train(epoch, keep_data=False)
    out_val = validate(epoch, keep_data=keep_data)

    if keep_data:
        loss_val, data_val = out_val
        st = time.time()
        # training_summaries(data,
        #                    epoch,
        #                    summary_writer)
        print(f"{time_str()} Wrote summary data to tensorboard in {time.time() - st:.2f}s.")

print("Finished training. Exiting.")

