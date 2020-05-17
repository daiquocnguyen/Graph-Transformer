#! /usr/bin/env python

import tensorflow as tf
import numpy as np
np.random.seed(123)
tf.compat.v1.set_random_seed(123)

import os
import time
import datetime
from model_unsup_gcn import GCN_graph_cls
import pickle as cPickle
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from scipy.sparse import coo_matrix
from util import *
import statistics
from sklearn.linear_model import LogisticRegression

# Parameters
# ==================================================

parser = ArgumentParser("GCN_Unsup", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')

parser.add_argument("--run_folder", default="../", help="")
parser.add_argument("--dataset", default="MUTAG", help="Name of the dataset.")
parser.add_argument("--learning_rate", default=0.0005, type=float, help="Learning rate")
parser.add_argument("--batch_size", default=2, type=int, help="Batch Size")
parser.add_argument("--num_epochs", default=50, type=int, help="Number of training epochs")
parser.add_argument("--saveStep", default=1, type=int, help="")
parser.add_argument("--allow_soft_placement", default=True, type=bool, help="Allow device soft device placement")
parser.add_argument("--log_device_placement", default=False, type=bool, help="Log placement of ops on devices")
parser.add_argument("--model_name", default='MUTAG', help="")
parser.add_argument("--dropout", default=0.5, type=float, help="Dropout")
parser.add_argument("--num_GNN_layers", default=2, type=int, help="Number of stacked layers")
parser.add_argument("--hidden_size", default=64, type=int, help="size of hidden layers")
parser.add_argument('--num_sampled', default=512, type=int, help='')
args = parser.parse_args()

print(args)

# Load data
print("Loading data...")

use_degree_as_tag = False
if args.dataset == 'COLLAB' or args.dataset == 'IMDBBINARY' or args.dataset == 'IMDBMULTI':
    use_degree_as_tag = True
graphs, num_classes = load_data(args.dataset, use_degree_as_tag)
feature_dim_size = graphs[0].node_features.shape[1]
graph_labels = np.array([graph.label for graph in graphs])
if "REDDIT" in args.dataset:
    feature_dim_size = 4

def get_Adj_matrix(batch_graph):
    edge_mat_list = []
    start_idx = [0]
    for i, graph in enumerate(batch_graph):
        start_idx.append(start_idx[i] + len(graph.g))
        edge_mat_list.append(graph.edge_mat + start_idx[i])

    Adj_block_idx = np.concatenate(edge_mat_list, 1)
    Adj_block_elem = np.ones(Adj_block_idx.shape[1])

    #self-loop
    num_node = start_idx[-1]
    self_loop_edge = np.array([range(num_node), range(num_node)])
    elem = np.ones(num_node)
    Adj_block_idx = np.concatenate([Adj_block_idx, self_loop_edge], 1)
    Adj_block_elem = np.concatenate([Adj_block_elem, elem], 0)

    Adj_block = coo_matrix((Adj_block_elem, Adj_block_idx), shape=(num_node, num_node))

    return Adj_block

def get_graphpool(batch_graph):
    start_idx = [0]
    # compute the padded neighbor list
    for i, graph in enumerate(batch_graph):
        start_idx.append(start_idx[i] + len(graph.g))

    idx = []
    elem = []
    for i, graph in enumerate(batch_graph):
        elem.extend([1] * len(graph.g))
        idx.extend([[i, j] for j in range(start_idx[i], start_idx[i + 1], 1)])

    elem = np.array(elem)
    idx = np.array(idx)

    graph_pool = coo_matrix((elem, (idx[:, 0], idx[:, 1])), shape=(len(batch_graph), start_idx[-1]))
    return graph_pool

graph_pool = get_graphpool(graphs)

def get_idx_nodes(selected_graph_idx):
    idx_nodes = [np.where(graph_pool.getrow(i).toarray()[0] == 1)[0] for i in selected_graph_idx]
    idx_nodes = np.reshape(np.concatenate(idx_nodes), (-1, 1))
    return idx_nodes

def get_batch_data(batch_graph):
    # features
    X_concat = np.concatenate([graph.node_features for graph in batch_graph], 0)
    if "REDDIT" in args.dataset:
        X_concat = np.tile(X_concat, feature_dim_size) #[1,1,1,1]
        X_concat = X_concat * 0.01

    X_concat = coo_matrix(X_concat)
    X_concat = sparse_to_tuple(X_concat)
    # adj
    Adj_block = get_Adj_matrix(batch_graph)
    Adj_block = sparse_to_tuple(Adj_block)

    num_features_nonzero = X_concat[1].shape
    return Adj_block, X_concat, num_features_nonzero

class Batch_Loader(object):
    def __call__(self):
        selected_idx = np.random.permutation(len(graphs))[:args.batch_size]
        batch_graph = [graphs[idx] for idx in selected_idx]
        Adj_block, X_concat, num_features_nonzero = get_batch_data(batch_graph)
        idx_nodes = get_idx_nodes(selected_idx)
        return Adj_block, X_concat, num_features_nonzero, idx_nodes
batch_nodes = Batch_Loader()

print("Loading data... finished!")
# Training
# ==================================================
with tf.Graph().as_default():
    session_conf = tf.compat.v1.ConfigProto(allow_soft_placement=args.allow_soft_placement, log_device_placement=args.log_device_placement)
    session_conf.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=session_conf)
    with sess.as_default():
        global_step = tf.Variable(0, name="global_step", trainable=False)
        unsup_gcn = GCN_graph_cls(feature_dim_size=feature_dim_size,
                      hidden_size=args.hidden_size,
                      num_GNN_layers=args.num_GNN_layers,
                      vocab_size=graph_pool.shape[1],
                      num_sampled=args.num_sampled,
                      )

        # Define Training procedure
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=args.learning_rate)
        grads_and_vars = optimizer.compute_gradients(unsup_gcn.total_loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        out_dir = os.path.abspath(os.path.join(args.run_folder, "../runs_GCN_UnSup", args.model_name))
        print("Writing to {}\n".format(out_dir))

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # Initialize all variables
        sess.run(tf.compat.v1.global_variables_initializer())
        graph = tf.compat.v1.get_default_graph()

        def train_step(Adj_block, X_concat, num_features_nonzero, idx_nodes):
            feed_dict = {
                unsup_gcn.Adj_block: Adj_block,
                unsup_gcn.X_concat: X_concat,
                unsup_gcn.num_features_nonzero: num_features_nonzero,
                unsup_gcn.dropout: args.dropout,
                unsup_gcn.input_y:idx_nodes
            }
            _, step, loss = sess.run([train_op, global_step, unsup_gcn.total_loss], feed_dict)
            return loss

        write_acc = open(checkpoint_prefix + '_acc.txt', 'w')
        max_acc = 0.0
        idx_epoch = 0
        num_batches_per_epoch = int((len(graphs) - 1) / args.batch_size) + 1
        for epoch in range(1, args.num_epochs+1):
            loss = 0
            for _ in range(num_batches_per_epoch):
                Adj_block, X_concat, num_features_nonzero, idx_nodes = batch_nodes()
                loss += train_step(Adj_block, X_concat, num_features_nonzero, idx_nodes)
                # current_step = tf.compat.v1.train.global_step(sess, global_step)
            print(loss)
            # It will give tensor object
            node_embeddings = graph.get_tensor_by_name('embedding/node_embeddings:0')
            node_embeddings = sess.run(node_embeddings)
            graph_embeddings = graph_pool.dot(node_embeddings)
            #
            acc_10folds = []
            for fold_idx in range(10):
                train_idx, test_idx = separate_data_idx(graphs, fold_idx)
                train_graph_embeddings = graph_embeddings[train_idx]
                test_graph_embeddings = graph_embeddings[test_idx]
                train_labels = graph_labels[train_idx]
                test_labels = graph_labels[test_idx]

                cls = LogisticRegression(tol=0.001)
                cls.fit(train_graph_embeddings, train_labels)
                ACC = cls.score(test_graph_embeddings, test_labels)
                acc_10folds.append(ACC)
                print('epoch ', epoch, ' fold ', fold_idx, ' acc ', ACC)

            mean_10folds = statistics.mean(acc_10folds)
            std_10folds = statistics.stdev(acc_10folds)
            print('epoch ', epoch, ' mean: ', str(mean_10folds*100), ' std: ', str(std_10folds*100))

            write_acc.write('epoch ' + str(epoch) + ' mean: ' + str(mean_10folds*100) + ' std: ' + str(std_10folds*100) + '\n')

        write_acc.close()