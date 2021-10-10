import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import TransformerEncoder, TransformerEncoderLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""@Dai Quoc Nguyen"""
"""Graph Transformer with Gated GNN"""
class GatedGT(nn.Module):
    def __init__(self, feature_dim_size, hidden_size, num_classes,
                 num_self_att_layers, num_GNN_layers, nhead, dropout, act=torch.relu):
        super(GatedGT, self).__init__()
        self.num_GNN_layers = num_GNN_layers
        self.emb_encode = nn.Linear(feature_dim_size, hidden_size)
        self.dropout_encode = nn.Dropout(dropout)
        self.gt_layers = torch.nn.ModuleList()
        for _layer in range(self.num_GNN_layers):
            encoder_layers = TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, dim_feedforward=hidden_size, dropout=0.5)  # Default batch_first=False (seq, batch, feature) for pytorch < 1.9.0
            self.gt_layers.append(TransformerEncoder(encoder_layers, num_self_att_layers))
        self.z0 = nn.Linear(hidden_size, hidden_size)
        self.z1 = nn.Linear(hidden_size, hidden_size)
        self.r0 = nn.Linear(hidden_size, hidden_size)
        self.r1 = nn.Linear(hidden_size, hidden_size)
        self.h0 = nn.Linear(hidden_size, hidden_size)
        self.h1 = nn.Linear(hidden_size, hidden_size)
        self.soft_att = nn.Linear(hidden_size, 1)
        self.ln = nn.Linear(hidden_size, hidden_size)
        self.act = act
        self.prediction = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def gatedGNN(self, x, adj):
        a = torch.matmul(adj, x)
        # update gate
        z0 = self.z0(a)
        z1 = self.z1(x)
        z = torch.sigmoid(z0 + z1)
        # reset gate
        r = torch.sigmoid(self.r0(a) + self.r1(x))
        # update embeddings
        h = self.act(self.h0(a) + self.h1(r * x))

        return h * z + x * (1 - z)

    def forward(self, inputs, adj, mask):
        x = inputs
        x = self.dropout_encode(x)
        x = self.emb_encode(x)
        x = x * mask
        for idx_layer in range(self.num_GNN_layers):
            x = torch.transpose(x, 0, 1)  # (seq, batch, feature) for pytorch transformer, pytorch < 1.9.0
            x = self.gt_layers[idx_layer](x)
            x = torch.transpose(x, 0, 1)  # (batch, seq, feature)
            x = x * mask
            x = self.gatedGNN(x, adj) * mask
        # soft attention
        soft_att = torch.sigmoid(self.soft_att(x))
        x = self.act(self.ln(x))
        x = soft_att * x * mask
        # sum and max pooling
        graph_embeddings = torch.sum(x, 1) + torch.amax(x, 1)
        graph_embeddings = self.dropout(graph_embeddings)
        prediction_scores = self.prediction(graph_embeddings)

        return prediction_scores

"""Graph Transformer with GCN"""
class TextGraphTransformer(nn.Module):
    def __init__(self, feature_dim_size, hidden_size, num_classes,
                 num_self_att_layers, num_GNN_layers, nhead, dropout, act=torch.relu):
        super(TextGraphTransformer, self).__init__()
        self.num_GNN_layers = num_GNN_layers
        self.emb_encode = nn.Linear(feature_dim_size, hidden_size)
        self.dropout_encode = nn.Dropout(dropout)
        self.gt_layers = torch.nn.ModuleList()
        self.gcn_layers = torch.nn.ModuleList()
        for _layer in range(self.num_GNN_layers):
            encoder_layers = TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, dim_feedforward=hidden_size, dropout=0.5)  # Default batch_first=False (seq, batch, feature) for pytorch < 1.9.0
            self.gt_layers.append(TransformerEncoder(encoder_layers, num_self_att_layers))
            self.gcn_layers.append(GraphConvolution(hidden_size, hidden_size, dropout))
        self.soft_att = nn.Linear(hidden_size, 1)
        self.ln = nn.Linear(hidden_size, hidden_size)
        self.act = act
        self.prediction = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, adj, mask): # inputs: (batch, seq, feature)
        x = inputs
        x = self.dropout_encode(x)
        x = self.emb_encode(x)
        x = x * mask
        for idx_layer in range(self.num_GNN_layers):
            x = torch.transpose(x, 0, 1) # (seq, batch, feature) for pytorch transformer, pytorch < 1.9.0
            x = self.gt_layers[idx_layer](x)
            x = torch.transpose(x, 0, 1) # (batch, seq, feature)
            x = x * mask
            x = self.gcn_layers[idx_layer](x, adj) * mask
        # soft attention
        soft_att = torch.sigmoid(self.soft_att(x))
        x = self.act(self.ln(x))
        x = soft_att * x * mask
        # sum and max pooling
        graph_embeddings = torch.sum(x, 1) + torch.amax(x, 1) 
        graph_embeddings = self.dropout(graph_embeddings)
        prediction_scores = self.prediction(graph_embeddings)
        return prediction_scores

"""New advanced TextGCN using Residual Connection"""
class TextGCN(nn.Module):
    def __init__(self, feature_dim_size, hidden_size, num_GNN_layers, num_classes, dropout, act=torch.relu):
        super(TextGCN, self).__init__()
        self.num_GNN_layers = num_GNN_layers
        self.gnnlayers = torch.nn.ModuleList()
        for layer in range(self.num_GNN_layers):
            if layer == 0:
                self.gnnlayers.append(GraphConvolution(feature_dim_size, hidden_size, dropout, act=act))
            else:
                self.gnnlayers.append(GraphConvolution(hidden_size, hidden_size, dropout, act=act))
        self.soft_att = nn.Linear(hidden_size, 1)
        self.ln = nn.Linear(hidden_size, hidden_size)
        self.act = act
        self.prediction = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, adj, mask):
        x = inputs
        for idx_layer in range(self.num_GNN_layers):
            if idx_layer == 0:
                x = self.gnnlayers[idx_layer](x, adj) * mask
            else: # Residual Connection
                x += self.gnnlayers[idx_layer](x, adj) * mask
        # soft attention
        soft_att = torch.sigmoid(self.soft_att(x))
        x = self.act(self.ln(x))
        x = soft_att * x * mask
        # sum and max pooling
        graph_embeddings = torch.sum(x, 1) + torch.amax(x, 1) 
        graph_embeddings = self.dropout(graph_embeddings)
        prediction_scores = self.prediction(graph_embeddings)

        return prediction_scores

""" Simple GCN layer, similar to https://arxiv.org/abs/1609.02907 """
class GraphConvolution(torch.nn.Module):
    def __init__(self, in_features, out_features, dropout, act=torch.relu, bias=False):
        super(GraphConvolution, self).__init__()
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.act = act
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight.size(0) + self.weight.size(1)))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        x = self.dropout(input)
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            output = output + self.bias
        return self.act(output)
