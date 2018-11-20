import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class FastEncoder(nn.Module):
    """Alternative to the MLP encoder. Never does feature transformations
    on the edges, so should be a bit faster.
    Feature transformations (via FC layers + relus and dropout) are done
    on the nodes.
    The node transformations are divided into 2 parts:
     - the first is a global transformation
     - the second is a separate transformation for every edge type required
       in the output.

    Edge logits (for each edge types) are calculated by the pairwise distance
    between nodes. Distance calculation is thus fundamental for correct functioning
    of this module, and we implemented many options (see `calc_distance`).
    """
    ALLOWED_DISTANCES = ["dot", "svm", "norm", "squared", "cosine"]

    def __init__(self, n_in, n_hid, n_out, do_prob, dist_type="dot"):
        """
        Args:
         - n_in : int
            Number of input features. This is typically the number of
            time-steps in each sample.
         - n_hid : List[int]
            A list of three different hidden sizes. The first and second are used
            for the first node transformation. The third is the output dimension.
         - n_out : int
            The number of edge types. This is the output dimension after distance
            calculation.
         - do_prob : float
            Dropout probability.
         - dist_type : str (in ALLOWED_DISTANCES)
            The function used for distance calculation. Recommended functions are
            "svm", "dot". "norm" and "cosine" are unlikely to work well (due to
            normalization issues??).

        Notes:
          The output dimension is crucial for memory reasons (we will have one
          vector of that size for each edge), so it may be good to have it small.
        """
        super(FastEncoder, self).__init__()

        if len(n_hid) != 3:
            raise ValueError("Number of hidden layer sizes must be equal to 3!")

        self.edge_types = n_out
        self.n_node_feat = n_hid[-1]
        self.dropout_prob = do_prob

        if dist_type not in FastEncoder.ALLOWED_DISTANCES:
            raise ValueError("Distance type must be one of %s"
                            % (FastEncoder.ALLOWED_DISTANCES))
        self.distance = dist_type

        self.node_fc1 = nn.Linear(n_in, n_hid[0])
        self.node_fc2 = nn.Linear(n_hid[0], n_hid[1])

        self.spec_fc = nn.ModuleList([nn.Linear(n_hid[1], n_hid[2]) for i in range(n_out)])
        self.out_fc = nn.ModuleList([nn.Linear(n_hid[2], n_hid[2]) for i in range(n_out)])

        self._init_distance()
        self._init_weights()

    def forward(self, inputs, adj):
        """Run the model for an input batch.
        Args:
         - inputs : tensor [B, N, T]
         - adj : sparse tensor [N, N]
            This is a fully-connected graph (number of edges `E = nnz(adj)`)
        Returns:
         - edge_predictions : tensor [B, E, F]
            The number of edges depends on the number of non-zero elements
            of `adj`.

        Notes:
          We define the dimensions used here: B is the batch size;
          N is the number of nodes; T is the number of time-steps;
          E is the number of edges and F is the number of edge types.
        """
        x = inputs

        ## Transform the node features
        x = F.dropout(F.relu(self.node_fc1(x)), p=self.dropout_prob)  # [B, N, H0]
        x = F.dropout(F.relu(self.node_fc2(x)), p=self.dropout_prob)  # [B, N, H1]

        # This gives us the number of edges (row: edge source, col: edge destination).
        row, col = adj._indices()

        edge_predictions = torch.empty(x.size(0), row.size(0), self.edge_types,
                                       dtype=torch.float32, device=x.device)
        ## Single branch for each output edge type
        for i in range(self.edge_types):
            xe_curr = F.dropout(F.relu(self.spec_fc[i](x)), p=self.dropout_prob) # [B, N, H2]
            xe_curr = self.out_fc[i](xe_curr) # [B, N, H3]
            esrc = xe_curr[:,row,:] # [B, E, H3]
            edst = xe_curr[:,col,:] # [B, E, H3]
            logits = self.calc_distance(esrc, edst) # [B, E]
            edge_predictions[:,:,i] = logits

        return edge_predictions # [B, E, F]

    def _init_distance(self):
        if self.distance == "svm":
            self.alpha = nn.Parameter(nn.init.normal_(
                torch.empty(self.n_node_feat, 1, dtype=torch.float32),
                std=0.02))
            self.bias = nn.Parameter(nn.init.uniform_(
                torch.empty(1, dtype=torch.float32)))
        elif self.distance == "squared":
            self.W = nn.Parameter(nn.init.xavier_normal_(
                torch.empty(self.n_node_feat, self.n_node_feat, dtype=torch.float32)))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def calc_distance(self, e1, e2):
        if self.distance == "norm":
            p = 2
            dist = torch.norm(e1 - e2, p=p, dim=-1)
            return dist

        if self.distance == "dot":
            # dot-product is not a distance but a similarity measure
            # (higher dot-product: higher similarity)
            sim = torch.mul(e1, e2).sum(2)
            return sim

        if self.distance == "cosine":
            sim = F.cosine_similarity(e1, e2, 2)
            return sim

        if self.distance == "svm":
            alpha = self.alpha
            bias = self.bias

            dist = torch.sub(e1, e2)
            proj_dist = torch.matmul(dist, alpha) + bias
            proj_dist = proj_dist.squeeze(2)

            return proj_dist

        if self.distance == "squared":
            W = self.W

            tempW = torch.matmul(e1, W)
            dist = tempW.mul(e2).sum(-1)

            return dist


class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm.
    https://github.com/ethanfetaya/NRI/blob/master/modules.py
    """

    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return self.batch_norm(x)


class MLPEncoder(nn.Module):
    def __init__(self, n_in, n_hid, n_out, do_prob=0.0, factor=True):
        """
        Args:
         n_in : int
            Number of dimensions of the input (timesteps * dimension of each point).
         n_hid : int
            Dimensionality of hidden layer in all the MLPs used.
         n_out : int
            Dimensionality of the output layer (corresponds to the number of
            edge types).
         do_prob : float
            Dropout probability
         factor : bool (default True)
            Whether or not to use the factor graph MLP encoder. Factor graph
            encoder uses two layers?
        """
        super(MLPEncoder, self).__init__()

        self.factor = factor

        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
        if self.factor:
            self.mlp4 = MLP(n_hid * 3, n_hid, n_hid, do_prob)
            print("Using factor graph MLP encoder.")
        else:
            self.mlp4 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
            print("Using MLP encoder.")
        self.fc_out = nn.Linear(n_hid, n_out)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def edge2node(self, x, adj):
        """Passes messages from edges to nodes according to connectivity.

        In practice the resulting node vectors are a sum of all edge vectors
        which (originate/terminate) at any given node. This is where one could
        place an attention mechanism.

        Args:
          x : tensor [batch, num_edges, dim]
            Input matrix
          adj : sparse tensor [num_atoms, num_atoms]
            Binary adjacency matrix.
        Returns:
          nodes : tensor [batch, num_atoms, dim]

        """
        row, col = adj._indices()
        # TODO: should aggregation be on row (edge origin) or col (edge destination)?
        incoming = torch.zeros(x.size(0), adj.size(0), x.size(2), dtype=torch.float32, device=x.device)
        incoming.index_add_(1, col, x)

        return incoming

    def node2edge(self, x, adj):
        """Passes messages from nodes `x` along all neighboring edges

        In practice the edge vectors (of size dim*2) are a concatenation
        of the two node vectors which connect any given edge.

        Args:
          x : tensor [batch, num_atoms, dim]
            Input matrix
          adj : sparse tensor [num_atoms, num_atoms]
            Binary adjacency matrix.
        Returns:
          edges : tensor [batch, num_edges, dim*2]
        """
        row, col = adj._indices()
        edges = torch.cat((x[:,row], x[:,col]), dim=2)
        return edges

    def forward(self, inputs, adj):
        """
        Args:
         inputs : tensor [num_sims, num_atoms, num_timesteps, num_dims]
            The input data
          adj : sparse tensor [num_atoms, num_atoms]
            Binary adjacency matrix.
        Returns:
         x : tensor [num_sims, n_edges, n_out]
        """
        # Input shape: [num_sims, num_atoms, num_timesteps, num_dims]
        x = inputs.view(inputs.size(0), inputs.size(1), -1)
        # New shape: [num_sims, num_atoms, num_timesteps*num_dims]

        x = self.mlp1(x)  # [num_sims, num_atoms, n_hid]

        x = self.node2edge(x, adj) # [num_sims, num_edges, n_hid*2]
        x = self.mlp2(x) # [num_sims, num_edges, n_hid]
        x_skip = x

        if self.factor:
            x = self.edge2node(x, adj) # [num_sims, num_nodes, n_hid]
            x = self.mlp3(x) # [num_sims, num_nodes, n_hid]
            x = self.node2edge(x, adj) # [num_sims, num_edges, n_hid*2]
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x) # [num_sims, num_edges, n_hid]
        else:
            x = self.mlp3(x)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)

        return self.fc_out(x) # [num_sims, num_edges, n_out]


class MLPDecoder(nn.Module):
    def __init__(
        self,
        n_in : int,
        n_edge_types : int,
        n_atoms : int,
        msg_hid : int,
        msg_out : int,
        n_hid : int,
        n_classes : int,
        dropout_prob : float):
        """
        Args:
         n_in : int
            Number of dimensions of the input (timesteps * dimension of each point).
         n_edge_types : int
            The number of different edge types that have been defined.
            The first edge type is assumed to take "no edge" meaning.
         msg_hid : int
            Latent dimension of the "messages"
         msg_out : int
            Output dimension of the "messages"
         rnn_hid : int
            Hidden layer size of the RNN
         n_hid : int
            Latent dimension of the FC layers after RNN (for output)
         n_classes : int
            Number of output classes.
        """
        super(MLPDecoder, self).__init__()

        self.dropout_prob = dropout_prob
        self.msg_out = msg_out

        # triu indices
        self.triu_indices = [torch.from_numpy(t).long() for t in np.triu_indices(n_atoms, k=1)]

        # Message passing
        self.msg_fc1 = nn.Linear(n_atoms, msg_hid)
        self.msg_fc2 = nn.Linear(msg_hid, msg_out)

        self.out_fc1 = nn.Linear(msg_out, n_hid)
        self.out_fc2 = nn.Linear(n_hid,   n_hid)
        self.out_fc3 = nn.Linear(n_hid,   n_classes)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def to(self, device):
        super().to(device)
        self.triu_indices[0] = self.triu_indices[0].to(device)
        self.triu_indices[1] = self.triu_indices[1].to(device)

    def forward(self, inputs, sparse_edges):
        """
        Args:
         inputs : tensor [batch_size, num_atoms, num_timesteps, num_dims]
            The original inputs
         sparse_edges : tensor [batch_size, num_edges, num_edge_types]
            The inferred edge types
         rel_rec : tensor [num_edges, num_atoms]
            Binary matrix telling where incoming edges are.
         rel_send : tensor [num_edges, num_atoms]
            Binary matrix telling where outgoing edges are.
        """

        B, N, T = inputs.size()
        Et = sparse_edges.size(2)

        out_adj_mats = torch.zeros(B, N, self.msg_out, dtype=torch.float, device=inputs.device)
        for i in range(1, Et):
            edges = torch.zeros(B, N, N, dtype=torch.float32, device=inputs.device)
            edges[:,self.triu_indices[0], self.triu_indices[1]] = sparse_edges[:,:,i]
            edges = edges + edges.transpose(1, 2)
            #edges = sparse_edges[:,:,i].view(-1, 423, 422)
            # Compress the reduced adjacency matrix
            edges = self.msg_fc1(edges)
            edges = F.relu(edges)
            edges = F.dropout(edges, p=self.dropout_prob)
            edges = self.msg_fc2(edges)
            edges = F.relu(edges)
            edges = F.dropout(edges, p=self.dropout_prob)

            # Inceasingly strong contributions
            out_adj_mats = out_adj_mats + edges * (i * 0.2)

        # agg [B*N, msg_out]
        # Prediction MLP
        pred = F.dropout(F.relu(self.out_fc1(out_adj_mats)), p=self.dropout_prob) # B x N x F
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob) # B x N x F'

        # For final classif we can either use a convolution, which aggregates
        # through all nodes or an attention layer.
        # Both should work fine! Attention layer easier to implement. For now just
        # mean pooling is okay.

        # Node aggregation
        pred = torch.mean(pred, dim=1) # B x F'
        pred_logit = self.out_fc3(pred) # B x O

        return pred_logit


class MLPReconstructionDecoder(nn.Module):
    def __init__(self, n_atoms, n_in_node, gnn_hid_list):
        super(MLPReconstructionDecoder, self).__init__()

        # GNN parameters
        all_gnn_sizes = [n_in_node] + gnn_hid_list
        gnn_weights = []
        for i in range(len(all_gnn_sizes) - 1):
            w = nn.Parameter(nn.init.xavier_normal_(
                torch.empty(all_gnn_sizes[i], all_gnn_sizes[i+1],
                            dtype=torch.float32)))
            gnn_weights.append(w)
        self.gnn_weights = nn.ModuleList(gnn_weights)

        # Output MLP Parameters
        gnn_hid = all_gnn_sizes[-1]
        self.out_fc1 = nn.Linear(gnn_hid, gnn_hid)
        self.out_fc2 = nn.Linear(gnn_hid, rnn_hid)
        self.out_fc3 = nn.Linear(gnn_hid, n_in_node)

        # triu indices
        self.triu_indices = [torch.from_numpy(t).long() for t in np.triu_indices(n_atoms, k=1)]

    def to(self, device):
        super().to(device)
        self.triu_indices[0] = self.triu_indices[0].to(device)
        self.triu_indices[1] = self.triu_indices[1].to(device)

    def mlp_step(self, inputs, sparse_edges):
        """
        Aggregate the original inputs according to the graph.
        Then concatenate with the original inputs, and make prediction about the output time-steps.

        Args:
         inputs : tensor [batch_size, num_atoms, tstamp_batch, num_dims]
        """

        B, N, BT, D = inputs.size()
        inputs = inputs.transpose(1,2).view(B*BT,N,D).contiguous()
        Et = sparse_edges.size(2)

        # Create edges
        edge_list = []
        for i in range(1, Et):
            edges = torch.zeros(B, N, N, dtype=torch.float32, device=inputs.device)
            edges[:,self.triu_indices[0], self.triu_indices[1]] = sparse_edges[:,:,i]
            edges = edges + edges.transpose(1, 2)
            edge_list.append(edges.repeat(BT, 1, 1))

        # GNN
        x = inputs
        for layer in range(len(self.gnn_weights)):
            new_x = torch.zeros(B, N, self.gnn_weights[layer].size(1), dtype=x.dtype, device=x.device)
            # Modify embedding
            x = torch.mm(x, self.gnn_weights[layer])
            # Aggregate embedding (batched)
            for i in range(1, Et):
                aggregated = torch.bmm(edge_list[i], x) # B x N x D
                new_x += aggregated

            x = F.dropout(F.relu(new_x), p=self.dropout_prob) # B x N x D

        # Skip connection
        aug_inputs = torch.cat([inputs, x], dim=-1)

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(aug_inputs)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)

        # Residual + reshape for output
        output = inputs + pred
        output = output.view(B, BT, N, D).transpose(2, 1).contiguous()

        return output

    def forward(self, inputs, sparse_edges):
        """Predict the next time-step of the input time-series using the inferred graph.

        Time-step predictions occur from fixed-distance starting time points
        one starting point every `self.pred_steps`. In parallel, from
        each starting point we predict the next step. The original time-series
        is only used for seeding the initial starting points. Then predictions
        are fed back into the model.

        Args:
         inputs : tensor [batch_size, num_atoms, num_timesteps, num_dims]
            The original inputs
         sparse_edges : tensor [batch_size, num_edges, num_edge_types]
            The inferred edge types
        """
        B, N, T, D = inputs.size()
        assert self.pred_steps <= T

        preds = []

        # The predictions are done in batches of size T // pred_steps (BT)
        BT = T // self.pred_steps
        initial_time = inputs[:,:,0::self.pred_steps,:]
        output = torch.zeros(B, N, BT*self.pred_steps, D, dtype=torch.float32).to(inputs.device)

        for step in range(0, pred_steps):
            initial_time = self.mlp_step(initial_time, sparse_edges) # B x N x BT x D
            output[:,:,step::self.pred_steps,:] = initial_time

        all_predictions = output[:, :, :(T - 1), :]

        return all_predictions


class RNNReconstructionDecoder(nn.Module):
    """Recurrent decoder module."""

    def __init__(self, n_atoms, n_in_node, rnn_hid, gnn_hid_list, pred_steps, do_prob=0.):
        super(RNNReconstructionDecoder, self).__init__()

        # GNN parameters
        all_gnn_sizes = [n_in_node] + gnn_hid_list
        gnn_weights = []
        for i in range(len(all_gnn_sizes) - 1):
            w = nn.Parameter(nn.init.xavier_normal_(
                torch.empty(all_gnn_sizes[i], all_gnn_sizes[i+1],
                            dtype=torch.float32)))
            gnn_weights.append(w)
        self.gnn_weights = nn.ModuleList(gnn_weights)

        self.hidden_r = nn.Linear(rnn_hid, rnn_hid, bias=False)
        self.hidden_i = nn.Linear(rnn_hid, rnn_hid, bias=False)
        self.hidden_h = nn.Linear(rnn_hid, rnn_hid, bias=False)

        self.input_r = nn.Linear(gnn_hid_list[-1], rnn_hid, bias=True)
        self.input_i = nn.Linear(gnn_hid_list[-1], rnn_hid, bias=True)
        self.input_n = nn.Linear(gnn_hid_list[-1], rnn_hid, bias=True)

        self.out_fc1 = nn.Linear(rnn_hid, rnn_hid)
        self.out_fc2 = nn.Linear(rnn_hid, rnn_hid)
        self.out_fc3 = nn.Linear(rnn_hid, n_in_node)

        self.dropout_prob = do_prob
        self.rnn_hid = rnn_hid
        self.pred_steps = pred_steps

        # triu indices
        self.triu_indices = [torch.from_numpy(t).long() for t in np.triu_indices(n_atoms, k=1)]

    def to(self, device):
        super().to(device)
        self.triu_indices[0] = self.triu_indices[0].to(device)
        self.triu_indices[1] = self.triu_indices[1].to(device)

    def rnn_step(self, inputs, sparse_edges, hidden):
        """
        Args:
         - inputs : tensor [batch_size, num_atoms, num_dims]
         - sparse_edges : tensor [batch_size, num_edges, num_edge_types]
         - hidden : tensor [batch_size, num_atoms, rnn_hid]
        """
        # Graphs for each edge type. This is just a conversion from upper triangular to
        # a full graph (symmetric).
        B, N, Di = inputs.size()

        edge_list = []
        for i in range(1, Et):
            edges = torch.zeros(B, N, N, dtype=torch.float32, device=inputs.device)
            edges[:,self.triu_indices[0], self.triu_indices[1]] = sparse_edges[:,:,i]
            edges = edges + edges.transpose(1, 2)
            edge_list.append(edges)

        # GNN
        x = inputs
        for layer in range(len(self.gnn_weights)):
            new_x = torch.zeros(B, N, self.gnn_weights[layer].size(1), dtype=x.dtype, device=x.device)
            # Modify embedding
            x = torch.mm(x, self.gnn_weights[layer])
            # Aggregate embedding (batched)
            for i in range(1, Et):
                aggregated = torch.bmm(edge_list[i], x) # B x N x D
                new_x += aggregated

            x = F.dropout(F.relu(new_x), p=self.dropout_prob) # B x N x D

        # GRU-style gated aggregation
        r = F.sigmoid(self.input_r(inputs) + self.hidden_r(x))
        i = F.sigmoid(self.input_i(inputs) + self.hidden_i(x))
        n = F.tanh(self.input_n(inputs) + r * self.hidden_h(x))
        hidden = (1 - i) * n + i * hidden # B x N x D

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(hidden)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred) # B x N x F

        # Predict position/velocity difference
        pred = inputs + pred

        return pred, hidden

    def forward(self, inputs, sparse_edges):
        """
        Args:
         inputs : tensor [batch_size, num_atoms, num_timesteps, num_dims]
            The original inputs
         sparse_edges : tensor [batch_size, num_edges, num_edge_types]
            The inferred edge types
        """

        B, N, T, D = inputs.size()
        assert self.pred_steps <= T

        # Initialize RNN hidden state
        rnn_hidden = torch.zeros(B, N, self.rnn_hid, dtype=torch.float32, requires_grad=True).to(inputs.device)

        all_predictions = []

        for step in range(0, T - 1):
            if step % self.pred_steps == 0:
                # Predict based on ground truth
                rnn_input = inputs[:, :, step, :]
            else:
                rnn_input = all_predictions[step - 1]

            pred, rnn_hidden = self.rnn_step(inputs, sparse_edges, rnn_hidden)
            # pred: B x N x D
            all_predictions.append(pred)

        # Stack all time-steps
        return torch.stack(all_predictions, dim=2)


class TimeseriesClassifier(nn.Module):
    def __init__(self, n_in_node, mlp_hid_list, n_classes, do_prob):
        super(SimpleClassifier, self).__init__()

        all_mlp_dims = mlp_hid_list + [n_classes]

        mlps = []
        in_dim = n_in_node
        for h_dim in all_mlp_dims:
            mlps.append(nn.Linear(in_dim, h_dim))
            in_dim = h_dim

        self.mlps = nn.ModuleList(mlps)
        self.dropout_prob = do_prob


    def forward(self, inputs):
        """
        Args:
         inputs : tensor [batch_size, num_atoms, num_timesteps, num_dims]
        Returns:
         logits : tensor [batch_size, num_classes]
        """

        B, N, T, D = X.size()

        # Flatten last 2 dimensions
        X = inputs.view(B, N, -1)

        for i in range(len(self.mlps) - 1):
            X = F.dropout(F.relu(self.mlps[i](X)), p=self.dropout_prob)

        # Node aggregation
        pred = torch.mean(X, dim=1) # B x F'
        pred_logit = self.mlps[-1](X)

        return pred_logit



class VAEWithClasses(nn.Module):
    def __init__(self, encoder, decoder, temp, hard, prior):
        super(VAEWithClasses, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.temp = temp
        self.hard = hard

        # Handle the prior
        assert sum(prior) == 1.0, "Edge prior doesn't sum to 1"
        self.log_prior = torch.tensor(np.log(prior)).float()
        self.log_prior = self.log_prior.unsqueeze(0).unsqueeze(0)

    def to(self, device):
        super().to(device)
        self.log_prior = self.log_prior.to(device)

    def forward(self, X, A, Y):
        """
        Args:
         X : tensor [num_sims, num_atoms, num_timesteps, num_dims]
            The input data
         A : sparse tensor [num_atoms, num_atoms]
            Binary adjacency matrix.
         Y : tensor [num_sims, ]
            The class labels for every sample.
        Returns:
         x : tensor [num_sims, n_edges, n_out]
        """
        num_atoms = X.size(1)

        # We save many tensors in `self` since the caller should be able to access
        # them for control over reporting / analysis.
        self.edges, self.prob = self.run_encoder(X, A)
        self.output = self.run_decoder(X, self.edges)

        loss_kl = self.kl_categorical(self.prob, self.log_prior, num_atoms)

        if int(torch.__version__.split(".")[1]) == 4:
            loss_rec = F.cross_entropy(self.output, Y, size_average=True)
        else:
            loss_rec = F.cross_entropy(self.output, Y, reduction="elementwise_mean")

        return {"KL": loss_kl, "Reconstruction": loss_rec}

    def run_encoder(self, X, A):
        logits = self.encoder(X, A) # B x E x Et
        edges = F.gumbel_softmax(logits.view(-1, logits.size(2)),
                                 tau=self.temp, hard=self.hard).view(logits.size())
        prob = F.softmax(logits, dim=-1)

        return edges, prob

    def run_decoder(self, X, E):
        return self.decoder(X, E)

    def kl_categorical(self, preds, log_prior, num_atoms, eps=1e-16):
        kl_div = preds * (torch.log(preds + eps) - log_prior)
        return kl_div.sum() / preds.size(0) / num_atoms

    def delete_saved(self):
        """Delete the data from `self` so it doesn't pollute gpu memory
        """
        pass


class VAEWithClassesReconstruction(VAEWithClasses):
    def __init__(self, encoder, decoder, classifier, temp, hard, prior):
        super(VAEWithClassesReconstruction, self).__init__(encoder, decoder, temp, hard, prior)

        self.classifier = classifier

    def run_classifier(self, X):
        return self.classifier(X)

    def forward(self, X, A, Y):
        """
        Args:
         X : tensor [num_sims, num_atoms, num_timesteps, num_dims]
            The input data
         A : sparse tensor [num_atoms, num_atoms]
            Binary adjacency matrix.
         Y : tensor [num_sims, ]
            The class labels for every sample.
        Returns:
        """
        num_atoms = X.size(1)

        self.edges, self.prob = self.run_encoder(X, A)
        self.rec_output = self.run_decoder(X, edges)
        self.cls_output = self.run_classifier(self.rec_output)

        loss_kl = self.kl_categorical(self.prob, self.log_prior, num_atoms)

        if int(torch.__version__.split(".")[1]) == 4:
            loss_rec = F.mse(self.rec_output, X, size_average=True)
            loss_class = F.cross_entropy(self.cls_output, Y, size_average=True)
        else:
            loss_rec = F.mse(self.rec_output, X, reduction="elementwise_mean")
            loss_class = F.cross_entropy(self.cls_output, Y, reduction="elementwise_mean")

        return {"KL": loss_kl, "Reconstruction": loss_rec, "Classification": loss_class}

