
"""

Data:
 - Large mini-batches consisting of multidimensional time-series (e.g. sleep epochs)

Model should reconstruct the data, while inferring a graph!

Encoder:
 - Encoder should generate a graph (for each mini-batch) which represents the connectivity within
    the mini-batch.

Encoder v1
 - Encoder should generate the initial state of a RNN, which is then used/evolved along with the graph
    to predict the reconstruction of the input time-series. In this case the latent variable z represents nothing in particular?

Encoder v2
 - Encoder should generate the state of a RNN (at every time-point). In this case, I have no clue how to derive the
    KL divergence. Possibilities: Use a factorized distribution for generating z (i.e. generate it 1 time-point at a time, like in https://arxiv.org/pdf/1803.02991.pdf)
    (or https://arxiv.org/pdf/1506.02216.pdf). The first reference actually also has the G but in a slightly different fashion.
    This will need many many RNNs.

Encoder v3
 - State-space model?? A paper introducing a reparametrization, not clear on the motives.


Decoder:
 - This is slightly more easy: take the (graph is not necessary) and the (z latent variables), and generate reconstruction.

"""


class EdgeNodeGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_steps, num_nodes, teacher_forcing=False):
        # TODO: Teacher forcing is only implemented for the training, where we actually can
        # teacher force.
        # For testing we need to unroll the GRU loop, lots of work.
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = 1
        self.num_steps = num_steps
        self.num_nodes = num_nodes
        self.teacher_forcing = teacher_forcing

        self.EW = nn.Parameter(nn.init.xavier_uniform_(
            torch.empty(input_size, hidden_size, dtype=torch.float32),
            gain=nn.init.calculate_gain("relu")))

        gru_input_size = hidden_size
        if self.teacher_forcing:
            gru_input_size = hidden_size + 1
        self.gru = nn.GRU(gru_input_size, 1, num_layers=1, bias=True)

    def forward(self, G, E, h_0, X=None):
        """
        Args:
         - G : batch_size, N*N, H
         - E : batch_size, N*N
         - h_0 : batch_size, hid_features
         - X : tensor [B, N, T*2]
        Returns:
         - X : batch_size, seq_length, hid_features
        """

        batch_size, num_edges, num_features = G.shape()

        if self.teacher_forcing:
            assert X is not None, (
                "Teacher forcing enabled but missing true values.")
            assert X.shape(2) == self.num_steps, (
                "Time-steps are incorrect for teacher forcing.")

        # Transform the edges
        G_z = torch.matmul(G, self.EW) # B, N*N, H'
        # Aggregate (e2n) from edges to nodes
        X_z = torch.zeros(batch_size, self.num_nodes, G_z.shape(2),
            dtype=torch.float32,
            device=G_z.device)
        X_z.index_add_(1, E, G_z) # B, N, H'

        gru_input = X_z.unsqueeze(0).expand(self.num_steps, -1, -1, -1) # T, B, N, H'
        if self.teacher_forcing:
            gru_input = torch.cat((gru_input,
                                   X.transpose(2, 0, 1).unsqueeze(3)), 3) # T, B, N, H'+1

        # Aggregate nodes with batch side:
        # nodes are passed into the GRU independently! This is like in NRI.
        gru_input = gru_input.view(self.num_steps, batch_size*self.num_nodes, -1) # T, B*N, H''

        # h_0 : [1, B*N, 1]
        gru_output, h_T = self.gru(gru_input, h_0)
        # gru_output : [T, B*N, 1]
        # h_T : [1, B*N, 1]
        gru_output = gru_output.view(self.num_steps, batch_size, self.num_nodes).transpose(1, 0, 2)
        # gru_output : [B, T, N]

        return gru_output



class EdgeNodeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.EWii = nn.Parameter(nn.init.normal_(
            torch.empty(input_size, hidden_size, dtype=torch.float32),
            std=0.02))
        self.EWif = nn.Parameter(nn.init.normal_(
            torch.empty(input_size, hidden_size, dtype=torch.float32),
            std=0.02))
        self.EWig = nn.Parameter(nn.init.normal_(
            torch.empty(input_size, hidden_size, dtype=torch.float32),
            std=0.02))
        self.EWio = nn.Parameter(nn.init.normal_(
            torch.empty(input_size, hidden_size, dtype=torch.float32),
            std=0.02))

        self.W_hi = nn.Parameter(nn.init.normal_(
            torch.empty(hidden_size, hidden_size, dtype=torch.float32),
            std=0.02))
        self.W_hf = nn.Parameter(nn.init.normal_(
            torch.empty(hidden_size, hidden_size, dtype=torch.float32),
            std=0.02))
        self.W_hg = nn.Parameter(nn.init.normal_(
            torch.empty(hidden_size, hidden_size, dtype=torch.float32),
            std=0.02))
        self.W_ho = nn.Parameter(nn.init.normal_(
            torch.empty(hidden_size, hidden_size, dtype=torch.float32),
            std=0.02))

        self.b_ii = nn.Parameter(nn.ones(hidden_size, dtype=torch.float32))
        self.b_if = nn.Parameter(nn.ones(hidden_size, dtype=torch.float32))
        self.b_ig = nn.Parameter(nn.ones(hidden_size, dtype=torch.float32))
        self.b_io = nn.Parameter(nn.ones(hidden_size, dtype=torch.float32))

        self.b_hi = nn.Parameter(nn.ones(hidden_size, dtype=torch.float32))
        self.b_hf = nn.Parameter(nn.ones(hidden_size, dtype=torch.float32))
        self.b_hg = nn.Parameter(nn.ones(hidden_size, dtype=torch.float32))
        self.b_ho = nn.Parameter(nn.ones(hidden_size, dtype=torch.float32))

    def e2n(self, G, E):
        out = torch.zeros(G.size(0), self.hidden_size, dtype=torch.float32, device=G.device)
        out.index_add_(1, E, G)
        return out

    def forward(self, G, E, h_0, c_0):
        """
        Args:
         - G : batch_size, input_features
         - E : batch_size, input_features
         - h_0 : num_layers, batch_size, hid_features
         - c_0 : num_layers, batch_size, hid_features
        Returns:
         - X : batch_size, seq_length, hid_features
         - h_T : num_layers, batch_size, hid_features
         - c_T : num_layers, batch_size, hid_features
        """

        # Transform the edges
        G_ii = torch.matmul(G, self.EWii)
        G_if = torch.matmul(G, self.EWif)
        G_ig = torch.matmul(G, self.EWig)
        G_io = torch.matmul(G, self.EWio)

        # Edge2Node operation
        X_ii = self.e2n(G_ii, E)
        X_if = self.e2n(G_if, E)
        X_ig = self.e2n(G_ig, E)
        X_io = self.e2n(G_io, E)

        h_t = h_0
        c_t = c_0
        for step in range(self.num_steps):
            i_t = F.sigmoid(X_ii + self.b_ii + torch.matmul(h_t, self.W_hi) + self.b_hi)
            f_t = F.sigmoid(X_if + self.b_if + torch.matmul(h_t, self.W_hf) + self.b_hf)
            g_t = F.tanh(X_ig + self.b_ig + torch.matmul(h_t, self.W_hg) + self.b_hg)
            o_t = F.sigmoid(W_io + self.b_io + torch.matmul(h_t, self.W_ho) + self.b_ho)
            c_t = torch.matmul(f_t, c_t) + torch.matmul(i_t, g_t)
            h_t = torch.matmul(o_t, F.tanh(c_t))

        return o_t, (h_t, c_t)


class Conv(torch.nn.Module):
    """
    A convolution with the option to be causal and use xavier initialization
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 dilation=1, bias=True, w_init_gain='linear', is_causal=False):
        super(Conv, self).__init__()
        self.is_causal = is_causal
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.conv = nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    dilation=dilation, bias=bias)

        nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        if self.is_causal:
            padding = (int((self.kernel_size - 1) * (self.dilation)), 0)
            signal = F.pad(signal, padding)
        return self.conv(signal)



class TimeLaggedAutoencoder(nn.Module):
    """
    Learn an autoencoder which predicts the next time-steps.
    Since the data may have lags greater than 1, modeling is based on previous-K time-steps.

    """
    def __init__(
        self,
        n_timesteps,
        n_residual_channels,
        max_dilation,
        kernel_size,
        desired_edge_size,
        n_enc_layers,
        teacher_forcing):
        """
        Args:
         - n_timesteps : int
            The number of time-steps for each input sample.
         - n_residual_channels : int
            The number of channels to use as intermediate representation in the
            WaveNet encoder (the input channels are 2).
         - max_dilation : int
            This controls the receptive field size of the WaveNet. TODO: find out
            how this correlates with the n_timesteps. Whether or not this particular
            dilation is ever reached depends on `n_enc_layers`.
         - kernel_size : int
            Size of the 1D kernel for WaveNet (original WaveNet used size 2)
         - desired_edge_size : int
            Compress the time dimension of each sample to approximately
            `desired_edge_size` dimensions. This should not be too large to
            avoid memory issues (quadratic in memory).
         - n_enc_layers : int
            The total desired number of layers in the WaveNet encoder.
         - teacher_forcing : bool
            Whether to use teacher forcing when training (and testing?) the decoder.
        """

        # Example configuration
        self.n_timesteps = n_timesteps
        self.n_residual_channels = n_residual_channels # C
        self.n_dilate_layers = n_enc_layers
        self.n_strided_layers = math.floor(math.log2(n_timesteps / desired_edge_size))

        assert self.n_strided_layers <= self.n_dilate_layers, "Too many strided layers."

        """ Encoder """
        self.expand_layer = Conv(2, self.n_residual_channels, w_init_gain="linear")
        self.dilate_layers = nn.ModuleList()
        loop_factor = math.floor(math.log2(max_dilation)) + 1
        for i in range(self.n_dilate_layers):
            dilation = 2 ** (i % loop_factor)
            stride = 1 if i >= self.n_strided_layers else 2

            in_layer = Conv(self.n_residual_channels, self.n_residual_channels*2,
                            kernel_size=kernel_size, stride=stride,
                            dilation=dilation, w_init_gain="tanh",
                            is_causal=True)
            self.dilate_layers.append(in_layer)
        self.conv_out = Conv(self.n_residual_channels, 2, bias=False, w_init_gain="relu")
        self.conv_end = Conv(2, 1, bias=False, w_init_gain="linear")

        """ Decoder """
        self.dec_gru = EdgeNodeGRU(
            num_enc_features,
            num_dec_features,
            n_timesteps,
            num_nodes,
            teacher_forcing)



    def forward(self, X):
        """
        Args:
         - X : tensor [B, N, T*2]
        Returns:
         - X_recon : tensor [B, N, T]
        """

        num_nodes = X.shape(1)

        # TODO: Creation of `adj_tensor` should be moved to init, or outside the class.
        triu_mat = np.triu(np.ones([num_nodes, num_nodes]), k=1)
        A = to_sparse(torch.tensor(triu_mat.astype(np.float32)).to(X.device))

        G = self.fwd_encode(X, A)
        X_rec = self.fwd_decode(X, G, A)

        return G, X_rec


    def fwd_encode(self, X, A):
        """
        Encode every time-point based
        Args:
         - X : tensor [B, N, T*2]
        Returns:
         - Z : tensor [B, N*N, H]
        """
        # Mix the N channels in pairwise fashion, and reduce the time dimension

        # Data time-dimension should be double that of reconstruction (i.e. we only reconstruct the second half.)
        # Input channels should be 2 (i.e. the 2 nodes for an edge).
        # Time-dimension reduction can be achieved through strided convolution. Hopefully the negative impact of this can be remedied using larger kernel size?

        batch_size, num_nodes, seq_len = X.shape()

        # Start `forward` method.
        row, col = A._indices()
        edges = torch.stack((X[:,row], X[:,col]), dim=2) # B, N*N, 2, T*2

        forward_input = edges.view(-1, 2, seq_len) # B*N*N, 2, T*2

        forward_input = self.expand_layer(forward_input) # B*N*N, C, T*2

        for i in range(self.n_dilate_layers):
            in_act = self.dilate_layers[i](forward_input) # B*N*N, C*2, T (//2)
            t_act = F.tanh(in_act[:, :self.n_residual_channels, :])
            s_act = F.sigmoid(in_act[:, self.n_residual_channels:, :])
            # This is a weird gating thing. Taken from WaveNet, but they also have many
            # more frills (e.g. skip and residual connections).
            forward_input = t_act * s_act # B*N*N, C, T (//2)

        output = F.relu(forward_input, True) # B*N*N, C, H
        output = self.conv_out(output) # B*N*N, 2, H
        output = F.relu(output, True)
        output = self.conv_end(output) # B*N*N, 1, H
        output = output.squeeze(1).view(batch_size, num_nodes*num_nodes, -1) # B, N*N, H

        # TODO: We should also output a `style` vector somehow.

        return output


    def fwd_decode(self, X, Z, A):
        """
        Args:
         - Z : tensor [B, N*N, H]
         - X : tensor [B, N, T*2]
        Returns:
         - X_recon : tensor [B, N, T]
        """

        # Here we must use a RNN.
        # For every node, we have a set of N-1 neighbors each of which has a H feature vector.
        # The LSTM is the one we define above.
        # If not a RNN, one could also use a series of e2n, n2e operations to increase H dimensionality.

        h_0 = X[:,:,0]
        row, col = A._indices()

        gru_out = self.dec_gru(Z, col, h_0)

        return gru_out


    def loss(self, X, X_recon, G):
        """
        Args:
         - X : tensor [B, N, T*2]
         - X_recon : tensor [B, N, T]
        Returns:
        """
        X_orig = X[:,:,1:]

        l = F.mse_loss(X_orig, X_recon)

        return l


