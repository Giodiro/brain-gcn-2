
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



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
    def __init__(self, n_in, n_hid, n_out, do_prob=0., factor=True):
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
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)

    def edge2node(self, x, rel_rec, rel_send):
        """Passes messages from edges to nodes according to connectivity
        Args:
         x : tensor [batch, num_edges, dim]
            Input matrix
         rel_rec : tensor [num_edges, num_atoms]
            Binary matrix telling where incoming edges are.
         rel_send : tensor [num_edges, num_atoms]
            Binary matrix telling where outgoing edges are.
            NOTE This is unused here.
        Returns:
         nodes : tensor [batch, num_atoms, dim]

        NOTE:
         Assumes that we have the same graph across all samples.
        """
        rel_rect = rel_rec.t()
        incoming = torch.empty(x.size(0), rel_rect.size(0), x.size(2))
        for b in range(x.size(0)):
            incoming[b] = torch.mm(rel_rect, x[b])
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        """Passes messages from nodes `x` along all neighboring edges
        Args:
         x : tensor [batch, num_atoms, dim]
            Input matrix
         rel_rec : tensor [num_edges, num_atoms]
            Binary matrix telling where incoming edges are.
         rel_send : tensor [num_edges, num_atoms]
            Binary matrix telling where outgoing edges are.
        Returns:
         edges : tensor [batch, num_edges, dim*2]

        NOTE:
         Assumes that we have the same graph across all samples.
        """
        # matmuls are bmm + logic to get the dimensions correct
        receivers = torch.empty(x.size(0), rel_rec.size(0), x.size(2), dtype=torch.float32, device=x.device, requires_grad=True)
        senders = torch.empty(x.size(0), rel_rec.size(0), x.size(2), dtype=torch.float32, device=x.device, requires_grad=True)
        for b in range(x.size(0)):
            receivers[b] = torch.mm(rel_rec, x[b])
            senders[b] = torch.mm(rel_send, x[b])
        edges = torch.cat([receivers, senders], dim=2)
        return edges

    def node2edgev2(self, x, adj):
        row, col = adj._indices()
        
        edges = torch.stack((x[row], x[col]), dim=2)
        return edges

    def forward(self, inputs, rel_rec, rel_send):
        """
        Args:
         inputs : tensor [num_sims, num_atoms, num_timesteps, num_dims]
            The input data
         rel_rec : tensor [num_edges, num_atoms]
            Binary matrix telling where incoming edges are.
         rel_send : tensor [num_edges, num_atoms]
            Binary matrix telling where outgoing edges are.

        Returns:
         x : tensor [num_sims, n_edges, n_out]
        """
        # Input shape: [num_sims, num_atoms, num_timesteps, num_dims]
        x = inputs.view(inputs.size(0), inputs.size(1), -1)
        # New shape: [num_sims, num_atoms, num_timesteps*num_dims]

        x = self.mlp1(x)  # [num_sims, num_atoms, n_hid]

        x = self.node2edgev2(x, rel_rec) # [num_sims, num_edges, n_hid*2]
        x = self.mlp2(x) # [num_sims, num_edges, n_hid]
        x_skip = x

        if self.factor:
            x = self.edge2node(x, rel_rec, rel_send) # [num_sims, num_nodes, n_hid]
            x = self.mlp3(x) # [num_sims, num_nodes, n_hid]
            x = self.node2edge(x, rel_rec, rel_send) # [num_sims, num_edges, n_hid*2]
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
        n_in_node : int,
        n_edge_types : int,
        msg_hid : int,
        msg_out : int,
        n_hid : int,
        rnn_hid : int,
        n_classes : int,
        dropout_prob : float):
        """
        Args:
         n_in_node : int
            The number of nodes (also indicated as N) in the input graphs.
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
        self.data_dim = 2

        self.agg_fc1 = nn.ModuleList([nn.Linear(250, msg_out) for i in range(n_edge_types)])
        self.msg_fc1 = nn.ModuleList([nn.Linear(2, msg_hid) for i in range(n_edge_types)])
        self.msg_fc2 = nn.ModuleList([nn.Linear(msg_hid, msg_out) for i in range(n_edge_types)])

        self.rnn = nn.GRU(input_size=n_in_node*(msg_out + 2),
                          hidden_size=rnn_hid,
                          num_layers=1,
                          batch_first=True,
                          dropout=dropout_prob)

        self.out_fc1 = nn.Linear(msg_out, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_classes)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, inputs, rel_type, rel_rec, rel_send):
        """
        Args:
         inputs : tensor [batch_size, num_atoms, num_timesteps, num_dims]
            The original inputs
         rel_type : tensor [batch_size, num_edges, num_edge_types]
            The inferred edge types
         rel_rec : tensor [num_edges, num_atoms]
            Binary matrix telling where incoming edges are.
         rel_send : tensor [num_edges, num_atoms]
            Binary matrix telling where outgoing edges are.
        """

        B, N, T = inputs.size()
        B, E, Et = rel_type.size()

        rel_type_A = rel_type.view(B, N, N, Et)
        X = inputs



        out_wv = torch.zeros(B, N, self.msg_out, dtype=torch.float32, deice=X.device, requires_grad=True)
        for i in range(1, Et):
            trans_in = self.agg_fc1[i](inputs)
            for b in range(B):
                out_b = torch.mm(rel_type_A[b,:,:,i:i+1], trans_in[b])
                out_b = F.relu(out_b)
                out_b = F.dropout(out_b, p=self.dropout_prob)
                out_wv[b] += out_b

        # Prediction MLP
        pred = F.dropout(F.relu(self.out_fc1(out_wv)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred_logit = self.out_fc3(pred)

        return pred_logit


        rel_type = rel_type.unsqueeze(1).expand([B, T, E, Et])
        #inputs = inputs.transpose(1, 2) # [batch, T, N, D]

        receivers = torch.empty(B, T, E, 1, dtype=torch.float32, device=inputs.device, requires_grad=True)
        senders = torch.empty(B, T, E, 1, dtype=torch.float32, device=inputs.device, requires_grad=True)
        for b in range(B):
            receivers[b] = torch.mm(rel_rec, inputs[b]).t().unsqueeze(2)
            senders[b] = torch.mm(rel_send, inputs[b]).t().unsqueeze(2)

        pre_msg = torch.cat((receivers, senders), dim=-1) # [B, T, E, 2*D]

        all_msgs = torch.zeros(B, T, E, self.msg_out).to(inputs.device)

        # Offset by one to exclude first edge which indicates `no edge`.
        for i in range(1, Et):
            msg = pre_msg * rel_type[:,:,:,i:i+1]

            msg = F.relu(self.msg_fc1[i](msg))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = F.relu(self.msg_fc2[i](msg))
            all_msgs += msg

        # Aggregate edge->nodes
        all_msgs = all_msgs.transpose(1, 2).view(B, E, -1)
        rel_rect = rel_rec.t()
        agg_msgs = torch.empty(B, T, N, all_msgs.size(-1), dtype=torch.float32, device=inputs.device, requires_grad=True)
        for b in range(B):
            agg_msgs[b] = torch.mm(rel_rect, all_msgs[b]).view(B, N, T, -1).transpose(2, 1)

        # Skip connection
        aug_inputs = torch.cat([inputs, agg_msgs], dim=-1) # [B, T, N, O+D]

        # Output RNN
        _, h_n = self.rnn(pred.view(B, T, -1)) # [1, B, Rh]
        h_n = h_n.squeeze(0) # [B, Rh]

        # Prediction MLP
        pred = F.dropout(F.relu(self.out_fc1(h_n)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred_logit = self.out_fc3(pred)

        return pred_logit





