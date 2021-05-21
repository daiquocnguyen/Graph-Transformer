import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class FullyConnectedGT(nn.Module):

    def __init__(self, feature_dim_size, ff_hidden_size, num_classes,
                 num_self_att_layers, dropout, num_U2GNN_layers):
        super(FullyConnectedGT, self).__init__()
        self.feature_dim_size = feature_dim_size
        self.ff_hidden_size = ff_hidden_size
        self.num_classes = num_classes
        self.num_self_att_layers = num_self_att_layers #Each U2GNN layer consists of a number of self-attention layers
        self.num_U2GNN_layers = num_U2GNN_layers
        self.lst_gnn = torch.nn.ModuleList()
        #
        self.u2gnn_layers = torch.nn.ModuleList()
        for _layer in range(self.num_U2GNN_layers): # nhead=1 because the size of input feature vectors is odd
            encoder_layers = TransformerEncoderLayer(d_model=self.feature_dim_size, nhead=1, dim_feedforward=self.ff_hidden_size, dropout=0.5)
            self.u2gnn_layers.append(TransformerEncoder(encoder_layers, self.num_self_att_layers))
            # if _layer != self.num_U2GNN_layers - 1:
            #     self.lst_gnn.append(GraphConvolution(self.feature_dim_size, self.feature_dim_size, act=torch.tanh))
            # else:
            self.lst_gnn.append(GraphConvolution(self.feature_dim_size, self.feature_dim_size, act=lambda x: x))

        # Linear function
        self.predictions = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        # self.predictions.append(nn.Linear(feature_dim_size, num_classes)) # For including feature vectors to predict graph labels???
        for _ in range(self.num_U2GNN_layers):
            self.predictions.append(nn.Linear(self.feature_dim_size, self.num_classes))
            self.dropouts.append(nn.Dropout(dropout))

    # Depend on the task, select forward or simple_forward
    def forward(self, Adj_block, node_features):
        prediction_scores = 0
        input_Tr = node_features
        for layer_idx in range(self.num_U2GNN_layers):
            # self-attention over all nodes
            input_Tr = torch.unsqueeze(input_Tr, 0)
            input_Tr = self.u2gnn_layers[layer_idx](input_Tr)
            input_Tr = torch.squeeze(input_Tr, 0)
            # take a sum and a linear transformation like GCN
            input_Tr = self.lst_gnn[layer_idx](input_Tr, Adj_block)
            # take a sum over all node representations to get graph representations
            graph_embedding = torch.sum(input_Tr, dim=0)
            graph_embedding = self.dropouts[layer_idx](graph_embedding)
            # Produce the final scores
            prediction_scores += self.predictions[layer_idx](graph_embedding)

        return prediction_scores

    def simple_forward(self, Adj_block, node_features):
        prediction_scores = 0
        input_Tr = node_features
        for layer_idx in range(self.num_U2GNN_layers):
            # take a sum over neighbors to obtain an input for the transformer self-attention network
            input_Tr = torch.spmm(Adj_block, input_Tr)
            # self-attention over all nodes
            input_Tr = torch.unsqueeze(input_Tr, 0)
            input_Tr = self.u2gnn_layers[layer_idx](input_Tr)
            input_Tr = torch.squeeze(input_Tr, 0)
            # take a sum over all node representations to get graph representations
            graph_embedding = torch.sum(input_Tr, dim=0)
            graph_embedding = self.dropouts[layer_idx](graph_embedding)
            # Produce the final scores
            prediction_scores += self.predictions[layer_idx](graph_embedding)

        return prediction_scores


""" GCN layer, similar to https://arxiv.org/abs/1609.02907 """
class GraphConvolution(torch.nn.Module):
    def __init__(self, in_features, out_features, act=torch.tanh, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.act = act
        self.bn = torch.nn.BatchNorm1d(out_features)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        output = self.bn(output)
        return self.act(output)

def label_smoothing(true_labels: torch.Tensor, classes: int, smoothing=0.1):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method
    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=true_labels.device)
        true_dist.fill_(smoothing / (classes - 1))
        true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)
    return true_dist