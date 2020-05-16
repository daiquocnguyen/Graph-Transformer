import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from sampled_softmax import  *

class TransformerU2GNN(nn.Module):

    def __init__(self, vocab_size, feature_dim_size, ff_hidden_size, sampled_num,
                 num_self_att_layers, num_U2GNN_layers, dropout, device):
        super(TransformerU2GNN, self).__init__()
        self.feature_dim_size = feature_dim_size
        self.ff_hidden_size = ff_hidden_size
        self.num_self_att_layers = num_self_att_layers #Each U2GNN layer consists of a number of self-attention layers
        self.num_U2GNN_layers = num_U2GNN_layers
        self.vocab_size = vocab_size
        self.sampled_num = sampled_num
        self.device = device
        #
        self.u2gnn_layers = torch.nn.ModuleList()
        for _ in range(self.num_U2GNN_layers):
            encoder_layers = TransformerEncoderLayer(d_model=self.feature_dim_size, nhead=1, dim_feedforward=self.ff_hidden_size, dropout=0.5) # embed_dim must be divisible by num_heads
            self.u2gnn_layers.append(TransformerEncoder(encoder_layers, self.num_self_att_layers))
        # Linear function
        self.dropouts = nn.Dropout(dropout)
        self.ss = SampledSoftmax(self.vocab_size, self.sampled_num, self.feature_dim_size*self.num_U2GNN_layers, self.device)

    def forward(self, X_concat, input_x, input_y):
        output_vectors = [] # should test output_vectors = [X_concat]
        input_Tr = F.embedding(input_x, X_concat)
        for layer_idx in range(self.num_U2GNN_layers):
            #
            output_Tr = self.u2gnn_layers[layer_idx](input_Tr)
            output_Tr = torch.split(output_Tr, split_size_or_sections=1, dim=1)[0]
            output_Tr = torch.squeeze(output_Tr, dim=1)
            #new input for next layer
            input_Tr = F.embedding(input_x, output_Tr)
            output_vectors.append(output_Tr)

        output_vectors = torch.cat(output_vectors, dim=1)
        output_vectors = self.dropouts(output_vectors)

        logits = self.ss(output_vectors, input_y)

        return logits

