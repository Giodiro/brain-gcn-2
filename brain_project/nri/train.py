

train_loader, valid_loader, test_loader = load_data(args.batch_size, args.suffix)


# Generate off-diagonal interaction graph
off_diag = np.ones([args.num_atoms, args.num_atoms]) - np.eye(args.num_atoms)

rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
rel_rec = torch.tensor(rel_rec).to(device)
rel_send = torch.tensor(rel_send).to(device)


# Encoder
encoder = MLPEncoder(args.timesteps * args.dims,
                     args.encoder_hidden,
                     args.edge_types,
                     args.encoder_dropout,
                     args.factor)
next(encoder.parameters()).to(device)

# Prior
prior = np.array([0.91, 0.03, 0.03, 0.03])
log_prior = torch.tensor(np.log(prior)).to(device)
log_prior = log_prior.unsqueeze(0).unsqueeze(0)

# Decoder
decoder = MLPDecoder(n_in_node=args.num_atoms,
                     n_edge_types=len(prior),
                     msg_hid=128,
                     msg_out=16,
                     n_hid=128,
                     rnn_hid=128,
                     n_classes=args.num_classes)
next(decoder.parameters()).to(device)

# Training
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay, gamma=args.gamma)



def train(epoch, best_val_loss):
    t = time.time()

    encoder.train()
    decoder.train()
    scheduler.step()
    for batch_idx, inputs in enumerate(train_loader):
        data = inputs["??"].to(device)
        labels = inputs["target"].to(device)

        optimizer.zero_grad()

        logits = encoder(data, rel_rec, rel_send)
        edges = gumbel_softmax(logits, tau=args.temp, hard=args.hard)
        prob = F.softmax(logits, dim=-1)


        output = decoder(data, edges, rel_rec, rel_send)

        # ... Loss calculation wrt target ... #

        loss_kl = kl_categorical(prob, log_prior, args.num_atoms)

        loss = loss_kl + loss_rec


        loss.backward()
        optimizer.step()
