#! /usr/bin/env python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(123)

import numpy as np
np.random.seed(123)
import time

from model_MPGNN import *
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

# Parameters
# ==================================================
parser = ArgumentParser("TextGNN", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument("--dataset", default="mr", help="Name of the dataset.")
parser.add_argument("--learning_rate", default=0.0005, type=float, help="Learning rate")
parser.add_argument("--batch_size", default=5, type=int, help="Batch Size")
parser.add_argument("--num_epochs", default=100, type=int, help="Number of training epochs")
parser.add_argument("--dropout", default=0.5, type=float, help="Dropout")
parser.add_argument("--num_GNN_layers", default=2, type=int, help="Number of hidden layers")
parser.add_argument("--hidden_size", default=32, type=int)
parser.add_argument("--model", default="GT", help="GCN, GT")
parser.add_argument("--nhead", default=2, type=int)
args = parser.parse_args()
print(args)
# Load data
train_adj, train_feature, train_y, val_adj, val_feature, val_y, test_adj, test_feature, test_y = load_data(args.dataset)
# Some preprocessing
print('loading training set')
train_adj, train_mask = preprocess_adj(train_adj)
train_feature = preprocess_features(train_feature)
print('loading validation set')
val_adj, val_mask = preprocess_adj(val_adj)
val_feature = preprocess_features(val_feature)
print('loading test set')
test_adj, test_mask = preprocess_adj(test_adj)
test_feature = preprocess_features(test_feature)
feature_dim_size = train_feature.shape[-1]
num_classes = np.max(train_y)+1

print(train_adj.shape, val_adj.shape, test_adj.shape)
print(train_y.shape, test_y.shape)
print(train_feature.shape, val_feature.shape, test_feature.shape)

print("Loading data... finished!")

if "GCN" in args.model:
    model = TextGCN(feature_dim_size=feature_dim_size,
                    hidden_size=args.hidden_size,
                    num_GNN_layers=args.num_GNN_layers,
                    num_classes=num_classes,
                    dropout=args.dropout).to(device)
else:
    model = TextGraphTransformer(feature_dim_size=feature_dim_size,
                    hidden_size=args.hidden_size,
                    num_classes=num_classes,
                    num_self_att_layers=1,
                    num_GNN_layers=args.num_GNN_layers,
                    nhead=args.nhead,
                    dropout=args.dropout).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
num_batches_per_epoch = int((train_y.shape[0] - 1) / args.batch_size) + 1
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_batches_per_epoch, gamma=0.1)

def train():
    model.train() # Turn on the train mode
    total_loss = 0.
    # Training step
    indices = np.arange(0, len(train_y))
    np.random.shuffle(indices)
    for start in range(0, len(train_y), args.batch_size):
        end = start + args.batch_size
        idx = indices[start:end]
        optimizer.zero_grad()
        prediction_scores = model(torch.from_numpy(train_feature[idx]).float().to(device),
                                  torch.from_numpy(train_adj[idx]).float().to(device),
                                  torch.from_numpy(train_mask[idx]).float().to(device))
        loss = criterion(prediction_scores, torch.from_numpy(train_y[idx]).long().to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # prevent the exploding gradient problem
        optimizer.step()
        total_loss += loss.item()

    return total_loss

def evaluate(tmp_feature, tmp_adj, tmp_mask, tmp_y):
    model.eval() # Turn on the evaluation mode
    total_loss = 0.
    with torch.no_grad():
        prediction_scores = model(torch.from_numpy(tmp_feature).float().to(device),
                                  torch.from_numpy(tmp_adj).float().to(device),
                                  torch.from_numpy(tmp_mask).float().to(device)).detach()
    predictions = prediction_scores.max(1, keepdim=False)[1]
    labels = torch.from_numpy(tmp_y).long().to(device)
    correct = predictions.eq(labels.view_as(predictions)).sum().cpu().item()
    tmp_acc = correct / float(tmp_feature.shape[0])

    return tmp_acc

"""main process"""
best_val = 0
best_acc = 0
best_epoch = 0
cost_loss = []
for epoch in range(1, args.num_epochs + 1):
    epoch_start_time = time.time()
    train_loss = train()
    cost_loss.append(train_loss)
    # Validation
    val_acc = evaluate(val_feature, val_adj, val_mask, val_y)
    # Test
    test_acc = evaluate(test_feature, test_adj, test_mask, test_y)

    print('| epoch {:3d} | time: {:5.2f}s | loss {:5.2f} | val acc {:5.2f} | test acc {:5.2f} | '.format(
                epoch, (time.time() - epoch_start_time), train_loss, val_acc*100, test_acc*100))

    if best_val <= val_acc:
        best_val = val_acc
        best_epoch = epoch
        best_acc = test_acc

print("Best acc at epoch ", best_epoch, " : ", best_acc)
