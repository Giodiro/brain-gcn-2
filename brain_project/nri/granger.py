




class Granger(nn.Module):
    def __init__(
        self,
        num_nodes,
        lstm_burnin,
        theta,
        enc_hidden,
        latent_size,
        dec_hidden):
        """
        Args:
         - num_nodes : int
            The number of nodes in the input dataset
         - lstm_burnin : int
            The number of time-steps to discard when decoding the time-series.
            This must be lower than the total number of time-steps in the data.
         - theta : float
            Hyperparameter weighting the regularization term in the loss.
         - enc_hidden : int
            Size of the hidden state of the encoder LSTM.
         - latent_size : int
            Dimensionality of the Gaussian latent variable (the HOW variable).
         - dec_hidden : int
            Dimensionality of the hidden state of the decoder LSTM.
        """
        super(self, Granger).__init__()

        self.lstm_burnin = lstm_burnin
        self.theta = theta

        # The pairwise relationship indices.
        pairwise = np.ones((num_nodes, num_nodes)) - np.diag(np.ones(num_nodes))
        pairwise_row, pairwise_col = pairwise.nonzero()
        self.pairwise_row = torch.from_numpy(pairwise_row).long()
        self.pairwise_col = torch.from_numpy(pairwise_col).long()

        # Encoder
        self.enc_lstm = nn.LSTM(
            input_size=2,
            hidden_size=enc_hidden,
            num_layers=2,
            bidirectional=True)

        self.mu_layer = nn.Linear(enc_hidden, latent_size, bias=False)
        self.lv_layer = nn.Linear(enc_hidden, latent_size, bias=False)
        self.lambda_layer = nn.Linear(enc_hidden, latent_size, bias=False)

        # Decoder
        self.dec_lstm = nn.LSTM(
            input_size=1,
            hidden_size=dec_hidden,
            num_layers=2,
            bidirectional=False)

        self.pred_mlp = nn.Sequential(
            nn.Linear(dec_hidden + latent_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(1, bias=True))


    def to(self, device):
        super(self, Granger).to(device)
        self.pairwise_col = self.pairwise_col.to(device)
        self.pairwise_row = self.pairwise_row.to(device)


    def forward(self, X):
        """
        Args:
         - X : [B, T, N]
        Returns:
         - relation_prediction : tensor [B, E, Tr]
         - how_much_sample : tensor [B, E]

        E: number of edges
        B: batch size
        N: nodes
        T: time-steps
        Z: dimensionality of the stochastic latent relationship variable
        H_d: dimensionality of the decoder LSTM hidden state.
        Tr: Number of time-steps after removing the first `lstm_burnin`.
        enc_hidden: Size of the hidden state of the encoder LSTM
        """

        batch_size, seq_len, num_nodes = X.shape()
        num_relations = len(self.pairwise_row)

        ## 1. Aggregate the time-series to obtain pairwise relations
        X = X.transpose(1, 2) # B, N, T
        P = torch.stack((X[:,self.pairwise_row], X[:,self.pairwise_col]), 3) # B, E, T, 2
        # Aggregate the pairwise relations into the batch dimension.
        P = P.view(-1, seq_len, 2) # B*E, T, 2

        ## 2. Pass the pairwise relations through a LSTM. The output
        ##    parametrizes the latent distribution
        _, (h_n, c_n) = self.enc_lstm(P.transpose(1, 0))
        # enc_out : [T, B*E, 2*enc_hidden]
        # h_n : [num_layers*num_directions, B*E, enc_hidden]
        # Concatenate the hidden representation at the last time-step of LSTM for all
        # layers and directions.
        h_n = h_n.transpose(0, 1).view(batch_size*num_relations, -1) # B*E, num_layers*num_directions*enc_hidden

        ## 3.a) Parametrize & Sample from latent distribution (G_HOW)
        edge_mu = self.mu_layer(h_n) # B*E, Z
        edge_lv = self.lv_layer(h_n) # B*E, Z

        eps = torch.randn_like(edge_mu)
        edge_sample = edge_mu + torch.exp(edge_lv.mul(0.5)) * eps # B*E, Z
        edge_sample = edge_sample.view(batch_size, num_relations, -1) # B, E, Z

        ## 3.b) Parametrize & Sample from latent distribution (G_HOW_MUCH)
        how_much_lamda = self.lambda_layer(h_n) # B*E, 1
        eps = torch.randn_like(how_much_lamda) + 1e-10
        how_much_sample = -torch.log(eps) / how_much_lamda # B*E, 1
        # Sigmoid to make sure it's between 0 and 1.
        how_much_sample = F.sigmoid(how_much_sample)
        how_much_sample = how_much_sample.view(batch_size, num_relations, -1).squeeze(2) # B, E

        ## 4. Run a decoding LSTM on every node, to obtain a weird
        ##    relationship predictor (node-based)
        node_X = (X.transpose(0, 1) # T, B, N
                   .view(seq_len, batch_size * num_nodes) # T, B*N
                   .unsqueeze(2)) # T, B*N, 1
        dec_out, (h_n, c_n) = self.dec_lstm(node_X)
        # dec_out : [T, B*N, H_d]
        dec_out = dec_out.view(T, B, N, H_d).permute(1, 2, 0, 3) # B, N, T, H_d
        dec_out = dec_out[:,:,:self.lstm_burnin] # B, N, Tr, H_d
        seq_len_trunc = seq_len - self.lstm_burnin

        ## 5. Aggregate the previous node-based data with the `edge_sample` variable
        ##    (which is pairwise). Aggregation is done on the `pairwise_row` variable
        ##    which represents the "origin" nodes for each relationship.
        pairwise_dec_out = dec_out[:,self.pairwise_row] # B, E, Tr, H_d
        pairwise_features = torch.cat([edge_sample.unsqueeze(2).expand(-1, -1, seq_len_trunc, -1),
                                       pairwise_dec_out]) # B, E, Tr, H_d+Z

        ## 5. Pass the relationship prediction concatenated with the latent
        ##    variable (G_HOW) through a MLP to obtain an estimate time-series
        ##    for each neighboring node.
        relation_prediction = self.pred_mlp(pairwise_features) # B, E, Tr, 1
        relation_prediction = relation_prediction.squeeze(3) # B, E, Tr

        ## 6. Return the estimate time-series for each relationship, and the
        ##    G_HOW_MUCH estimate.
        return relation_prediction, how_much_sample

    def loss(self, relation_prediction, how_much_sample, true_timeseries):
        """
        Args:
         - relation_prediction : tensor [B, E, Tr]
         - how_much_sample : tensor [B, E]
         - true_timeseries : tensor [B, T, N]
        Returns:
         - mean_loss : scalar
        """

        # Here we aggregate by the `pairwise_col` variable which is the destination
        # node.
        true_timeseries = true_timeseries.transpose(1, 2) # B, N, T
        true_timeseries = true_timeseries[:,self.pairwise_col,:self.lstm_burnin] # B, E, Tr

        loss = (true_timeseries -
                (how_much_sample * true_timeseries + (1 - how_much_sample) * relation_prediction))
        loss = torch.mean(loss, 2) # B, E
        regularizer = self.theta * how_much_sample # B, E

        reg_loss = loss + regularizer

        mean_loss = torch.mean(reg_loss)
        return mean_loss





