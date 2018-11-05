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
        # self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
        if self.factor:
            self.mlp4 = MLP(n_hid * 3, n_hid, n_hid, do_prob)
            print("Using factor graph MLP encoder.")
        else:
            #self.mlp4 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
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
            #x = self.mlp3(x) # [num_sims, num_nodes, n_hid]
            x = self.node2edge(x, adj) # [num_sims, num_edges, n_hid*2]
            x = self.mlp2(x) # [num_sims, num_edges, n_hid]
            #x = torch.cat((x, x_skip), dim=2)  # Skip connection
            #x = self.mlp4(x) # [num_sims, num_edges, n_hid]
        else:
            #x = self.mlp3(x)
            #x = torch.cat((x, x_skip), dim=2)  # Skip connection
            #x = self.mlp4(x)
            pass

        return self.fc_out(x) # [num_sims, num_edges, n_out]



class MLPDecoder(nn.Module):
    def __init__(
        self,
        n_in : int,
        n_edge_types : int,
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

        # Message passing
        self.msg_fc1 = nn.Linear(422, msg_hid)
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

    def forward(self, inputs, sparse_edges):
        """
        Args:
         inputs : tensor [batch_size, num_atoms, num_timesteps, num_dims]
            The original inputs
         sparse_edges : List[sparse tensor [N x N]]
            The inferred edge types
         rel_rec : tensor [num_edges, num_atoms]
            Binary matrix telling where incoming edges are.
         rel_send : tensor [num_edges, num_atoms]
            Binary matrix telling where outgoing edges are.
        """

        B, N, T = inputs.size()
        Et = sparse_edges.size(2)

        out_adj_mats = torch.zeros(B, 423, self.msg_out, dtype=torch.float, device=sparse_edges.device)
        for i in range(1, Et):
            edges = sparse_edges[:,:,i].view(-1, 423, 422)
            # Compress the reduced adjacency matrix
            edges = self.msg_fc1(edges)
            edges = F.relu(edges)
            edges = F.dropout(edges, p=self.dropout_prob)
            edges = self.msg_fc2(edges)
            edges = F.relu(edges)
            edges = F.dropout(edges, p=self.dropout_prob)

            # Inceasingly strong contributions
            out_adj_mats = out_adj_mats + edges * (i * 0.2)

        # For final classif we can either use a convolution, which aggregatest through all nodes or an attention layer.
        # Both should work fine! Attention layer easier to implement.

        # agg [B*N, msg_out]
        # Prediction MLP
        pred = F.dropout(F.relu(self.out_fc1(out_adj_mats)), p=self.dropout_prob) # B x N x F
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob) # B x N x F'

        # Node aggregation
        pred = torch.mean(pred, dim=1) # B x F'
        pred_logit = self.out_fc3(pred) # B x O

        return pred_logit

