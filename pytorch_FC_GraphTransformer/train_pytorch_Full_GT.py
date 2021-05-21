#! /usr/bin/env python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(123)

import numpy as np
np.random.seed(123)
import time

from pytorch_FC_GT import *
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from scipy.sparse import coo_matrix
from util import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

# Parameters
# ==================================================

parser = ArgumentParser("U2GNN", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')

parser.add_argument("--run_folder", default="../", help="")
parser.add_argument("--dataset", default="PTC", help="Name of the dataset.")
parser.add_argument("--learning_rate", default=0.0005, type=float, help="Learning rate")
parser.add_argument("--num_epochs", default=50, type=int, help="Number of training epochs")
parser.add_argument("--model_name", default='PTC', help="")
parser.add_argument("--dropout", default=0.5, type=float, help="")
parser.add_argument("--num_hidden_layers", default=1, type=int, help="")
parser.add_argument("--num_timesteps", default=1, type=int, help="Timestep T ~ Number of self-attention layers within each U2GNN layer")
parser.add_argument("--ff_hidden_size", default=256, type=int, help="The hidden size for the feedforward layer")
parser.add_argument('--fold_idx', type=int, default=1, help='The fold index. 0-9.')
args = parser.parse_args()

print(args)

# Load data
print("Loading data...")

use_degree_as_tag = False
if args.dataset == 'COLLAB' or args.dataset == 'IMDBBINARY' or args.dataset == 'IMDBMULTI':
    use_degree_as_tag = True
graphs, num_classes = load_data(args.dataset, use_degree_as_tag)
# graph_labels = np.array([graph.label for graph in graphs])
# train_idx, test_idx = separate_data_idx(graphs, args.fold_idx)
train_graphs, test_graphs = separate_data(graphs, args.fold_idx)
feature_dim_size = graphs[0].node_features.shape[1]
print(feature_dim_size)
if "REDDIT" in args.dataset:
    feature_dim_size = 4

def get_Adj_matrix(graph):
    Adj_block_idx = torch.LongTensor(graph.edge_mat)
    Adj_block_elem = torch.ones(Adj_block_idx.shape[1])

    num_node = len(graph.g)
    self_loop_edge = torch.LongTensor([range(num_node), range(num_node)])
    elem = torch.ones(num_node)
    Adj_block_idx = torch.cat([Adj_block_idx, self_loop_edge], 1)
    Adj_block_elem = torch.cat([Adj_block_elem, elem], 0)

    Adj_block = torch.sparse.FloatTensor(Adj_block_idx, Adj_block_elem, torch.Size([num_node, num_node]))

    return Adj_block.to(device) # should use the Laplacian re-normalized adjacency matrix like in GCN?

def get_data(graph):
    node_features = graph.node_features
    if "REDDIT" in args.dataset:
        node_features = np.tile(node_features, feature_dim_size) #[1,1,1,1]
        node_features = node_features * 0.01
    node_features = torch.from_numpy(node_features).to(device)
    Adj_block = get_Adj_matrix(graph)
    graph_label = np.array([graph.label])
    return Adj_block, node_features, torch.from_numpy(graph_label).to(device)

# Adj_block, node_features, graph_label = get_data(train_graphs[1])
# print(Adj_block)
# print(node_features)
# print(graph_label)

print("Loading data... finished!")

model = FullyConnectedGT(feature_dim_size=feature_dim_size, ff_hidden_size=args.ff_hidden_size,
                        num_classes=num_classes, dropout=args.dropout,
                        num_self_att_layers=args.num_timesteps,
                        num_U2GNN_layers=args.num_hidden_layers).to(device)

def cross_entropy(pred, soft_targets): # use nn.CrossEntropyLoss if not using soft labels in Line 159
    logsoftmax = nn.LogSoftmax(dim=1)
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))
# criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

def train():
    model.train() # Turn on the train mode
    total_loss = 0.
    idxs = np.arange(len(train_graphs))
    np.random.shuffle(idxs)
    for idx in idxs:
        Adj_block, node_features, graph_label = get_data(train_graphs[idx]) # one graph per step, i.e., bs=1
        graph_label = label_smoothing(graph_label, num_classes)
        optimizer.zero_grad()
        prediction_score = model(Adj_block, node_features)
        # loss = criterion(prediction_scores, graph_labels)
        loss = cross_entropy(torch.unsqueeze(prediction_score, 0), graph_label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # prevent the exploding gradient problem
        optimizer.step()
        total_loss += loss.item()

    return total_loss

def evaluate():
    model.eval() # Turn on the evaluation mode
    total_loss = 0.
    with torch.no_grad():
        # evaluating
        prediction_output = []
        for i in range(0, len(test_graphs)):
            Adj_block, node_features, graph_label = get_data(test_graphs[i])
            prediction_score = model(Adj_block, node_features).detach()
            prediction_output.append(torch.unsqueeze(prediction_score, 0))
    prediction_output = torch.cat(prediction_output, 0)
    predictions = prediction_output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    correct = predictions.eq(labels.view_as(predictions)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))
    return acc_test

"""main process"""
import os
out_dir = os.path.abspath(os.path.join(args.run_folder, "../runs_pytorch_FC_GraphTransformer", args.model_name))
print("Writing to {}\n".format(out_dir))
# Checkpoint directory
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
write_acc = open(checkpoint_prefix + '_acc.txt', 'w')

cost_loss = []
for epoch in range(1, args.num_epochs + 1):
    epoch_start_time = time.time()
    train_loss = train()
    cost_loss.append(train_loss)
    acc_test = evaluate()
    print('| epoch {:3d} | time: {:5.2f}s | loss {:5.2f} | test acc {:5.2f} | '.format(
                epoch, (time.time() - epoch_start_time), train_loss, acc_test*100))

    write_acc.write('epoch ' + str(epoch) + ' fold ' + str(args.fold_idx) + ' acc ' + str(acc_test*100) + '%\n')

write_acc.close()
