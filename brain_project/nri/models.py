import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter_add


class FastEncoder(nn.Module):

    def __init__(self, n_in, n_hid, n_out, do_prob):
        super(FastEncoder, self).__init__()

        self.edge_types = n_out
        self.dropout_prob = do_prob

        self.node_fc1 = nn.Linear(n_in, n_hid)
        self.node_fc2 = nn.Linear(n_hid, n_hid)

        self.spec_fc = nn.ModuleList([nn.Linear(n_hid, n_hid) for i in range(n_out)])
        self.out_fc = nn.ModuleList([nn.Linear(n_hid, n_hid) for i in range(n_out)])


    def forward(self, inputs, adj):
        # Predict edge logits by using node similarity after a NN branch (1 for each edge)

        # Input shape: [num_sims, num_atoms, num_timesteps, num_dims]
        x = inputs.view(inputs.size(0), inputs.size(1), -1)
        # New shape: [num_sims, num_atoms, num_timesteps*num_dims]

        ## Transform the node features
        x = F.dropout(F.relu(self.node_fc1(x)), p=self.dropout_prob)  # [num_sims, num_atoms, n_hid]
        x = F.dropout(F.relu(self.node_fc2(x)), p=self.dropout_prob)  # [num_sims, num_atoms, n_hid]

        row, col = adj._indices()

        edge_predictions = torch.empty(x.size(0), row.size(0), self.edge_types,
                                       dtype=torch.float32, device=x.device)
        ## Single branch for each output edge type
        for i in range(self.edge_types):
            xe_curr = F.dropout(F.relu(self.spec_fc[i](x)), p=self.dropout_prob)
            xe_curr = self.out_fc[i](xe_curr) # B x N x F
            esrc = xe_curr[:,row,:] # B x E x F
            edst = xe_curr[:,col,:] # B x E x F
            logits = self.calc_distance(esrc, edst, "dot")
            edge_predictions[:,i] = logits

        return edge_predictions


    def calc_distance(self, e1, e2, dist_type):
        if dist_type == "norm":
            p = 2
            dist = torch.norm(e1 - e2, p=p, dim=-1)
            return dist

        if dist_type == "dot":
            # dot-product is not a distance but a similarity measure
            # (higher dot-product: higher similarity)
            sim = torch.mul(e1, e2).sum(2)
            return sim

        if dist_type == "cosine":
            sim = F.cosine_similarity(e1, e2, 2)
            return sim

        if dist_type == "svm":
            alpha = self.model.alpha
            bias = self.model.bias

            dist = torch.sub(e1, e2)
            proj_dist = torch.matmul(dist, alpha) + bias

            return proj_dist

        if dist_type == "squared":
            W = self.model.W

            tempW = torch.mm(e1, W)
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
        incoming = scatter_add(x, col, dim=1, dim_size=adj.size(0))
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

