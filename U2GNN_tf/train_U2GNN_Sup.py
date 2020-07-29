#! /usr/bin/env python

import tensorflow as tf
import numpy as np
np.random.seed(123456789)
tf.compat.v1.set_random_seed(123456789)

import os
import time
import datetime
from model_U2GNN_Sup_multi import U2GNN
import pickle as cPickle
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from scipy.sparse import coo_matrix
from util import *
import statistics

# Parameters
# ==================================================

parser = ArgumentParser("U2GNN", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')

parser.add_argument("--run_folder", default="../", help="")
parser.add_argument("--dataset", default="PTC", help="Name of the dataset.")
parser.add_argument("--embedding_dim", default=2, type=int, help="Dimensionality of character embedding")
parser.add_argument("--learning_rate", default=0.0005, type=float, help="Learning rate")
parser.add_argument("--batch_size", default=4, type=int, help="Batch Size")
parser.add_argument("--idx_time", default=1, type=int, help="")
parser.add_argument("--num_epochs", default=50, type=int, help="Number of training epochs")
parser.add_argument("--saveStep", default=1, type=int, help="")
parser.add_argument("--allow_soft_placement", default=True, type=bool, help="Allow device soft device placement")
parser.add_argument("--log_device_placement", default=False, type=bool, help="Log placement of ops on devices")
parser.add_argument("--model_name", default='PTC', help="")
parser.add_argument('--num_sampled', default=512, type=int, help='')
parser.add_argument("--dropout_keep_prob", default=1.0, type=float, help="Dropout keep probability")
parser.add_argument("--num_timesteps", default=3, type=int, help="Number of attention layers in Transformer. The number T of timesteps in Universal Transformer")
parser.add_argument("--num_heads", default=1, type=int, help="Number of attention heads within each attention layer")
parser.add_argument("--ff_hidden_size", default=1024, type=int, help="The hidden size for the feedforward layer")
parser.add_argument("--num_neighbors", default=16, type=int, help="")
parser.add_argument('--degree_as_tag', action="store_false", help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
parser.add_argument('--fold_idx', type=int, default=1, help='the index of fold in 10-fold validation. 0-9.')
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
num_nodes = sum([len(graph.g) for graph in graphs])
hparams_batch_size = int(num_nodes/len(graphs)) + 1
print(num_nodes, hparams_batch_size)
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

    Adj_block_idx_row = Adj_block_idx[0,:]
    Adj_block_idx_cl = Adj_block_idx[1,:]

    # Adj_block = coo_matrix((Adj_block_elem, (Adj_block_idx_row, Adj_block_idx_cl)), shape=(start_idx[-1], start_idx[-1])).toarray()

    return Adj_block_idx_row, Adj_block_idx_cl

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
    idx_row = idx[:,0]
    idx_cl = idx[:,1]

    graph_pool = coo_matrix((elem, (idx_row, idx_cl)), shape=(len(batch_graph), start_idx[-1]))
    # return idx_row, idx_cl
    return graph_pool

def get_batch_data(batch_graph):
    X_concat = np.concatenate([graph.node_features for graph in batch_graph], 0)
    if "REDDIT" in args.dataset:
        X_concat = np.tile(X_concat, feature_dim_size) #[1,1,1,1]
        X_concat = X_concat * 0.01

    graph_pool = get_graphpool(batch_graph)
    graph_pool = sparse_to_tuple(graph_pool)

    Adj_block_idx_row, Adj_block_idx_cl = get_Adj_matrix(batch_graph)
    dict_Adj_block = {}
    for i in range(len(Adj_block_idx_row)):
        if Adj_block_idx_row[i] not in dict_Adj_block:
            dict_Adj_block[Adj_block_idx_row[i]] = []
        dict_Adj_block[Adj_block_idx_row[i]].append(Adj_block_idx_cl[i])

    input_neighbors = []
    for input_node in range(X_concat.shape[0]):
        if input_node in dict_Adj_block:
            input_neighbors.append([input_node] + list(np.random.choice(dict_Adj_block[input_node], args.num_neighbors, replace=True)))
        else:
            input_neighbors.append([input_node for _ in range(args.num_neighbors+1)])
    input_x = np.array(input_neighbors)

    graph_labels = np.array([graph.label for graph in batch_graph])
    one_hot_labels = np.zeros((graph_labels.size, num_classes))
    one_hot_labels[np.arange(graph_labels.size), graph_labels] = 1

    return input_x, graph_pool, X_concat, one_hot_labels

class Batch_Loader(object):
    def __call__(self):
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]
        batch_graph = [train_graphs[idx] for idx in selected_idx]
        input_x, graph_pool, X_concat, one_hot_labels = get_batch_data(batch_graph)
        return input_x, graph_pool, X_concat, one_hot_labels
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
        u_gan = U2GNN(num_self_att_layers=args.num_timesteps,
                      hparams_batch_size=hparams_batch_size,
                      feature_dim_size=feature_dim_size,
                      ff_hidden_size=args.ff_hidden_size,
                      seq_length=args.num_neighbors+1,
                      num_classes=num_classes,
                      num_U2GNN_layers=1
                  )

        # Define Training procedure
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=args.learning_rate)
        grads_and_vars = optimizer.compute_gradients(u_gan.total_loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        out_dir = os.path.abspath(os.path.join(args.run_folder, "../runs_U2GNN_Sup", args.model_name))
        print("Writing to {}\n".format(out_dir))

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        graph = tf.get_default_graph()

        def train_step(input_x, graph_pool, X_concat, one_hot_labels):
            feed_dict = {
                u_gan.input_x: input_x,
                u_gan.graph_pool: graph_pool,
                u_gan.X_concat: X_concat,
                u_gan.one_hot_labels: one_hot_labels,
                u_gan.dropout_keep_prob: args.dropout_keep_prob
            }
            _, step, loss = sess.run([train_op, global_step, u_gan.total_loss], feed_dict)
            return loss

        def eval_step(input_x, graph_pool, X_concat, one_hot_labels):
            feed_dict = {
                u_gan.input_x: input_x,
                u_gan.graph_pool: graph_pool,
                u_gan.X_concat: X_concat,
                u_gan.one_hot_labels: one_hot_labels,
                u_gan.dropout_keep_prob: 1.0
            }
            _, acc = sess.run([global_step, u_gan.accuracy], feed_dict)
            return acc

        write_acc = open(checkpoint_prefix + '_acc.txt', 'w')
        num_batches_per_epoch = int((len(train_graphs) - 1) / args.batch_size) + 1
        for epoch in range(1, args.num_epochs + 1):
            loss = 0
            for _ in range(num_batches_per_epoch):
                input_x, graph_pool, X_concat, one_hot_labels = batch_nodes()
                loss += train_step(input_x, graph_pool, X_concat, one_hot_labels)
                # current_step = tf.compat.v1.train.global_step(sess, global_step)
            print(loss)

            acc_output = []
            # evaluating
            idx = np.arange(len(test_graphs))
            for i in range(0, len(test_graphs), args.batch_size):
                sampled_idx = idx[i:i + args.batch_size]
                if len(sampled_idx) == 0:
                    continue
                batch_test_graphs = [test_graphs[j] for j in sampled_idx]
                test_input_x, test_graph_pool, test_X_concat, test_one_hot_labels = get_batch_data(batch_test_graphs)
                acc_output.append(eval_step(test_input_x, test_graph_pool, test_X_concat, test_one_hot_labels))

            final_acc = sum(acc_output) / float(len(test_graphs))*100.0

            print('epoch ' + str(epoch) + ' fold ' + str(args.fold_idx) + ' acc ' + str(final_acc) + '%')

            write_acc.write('epoch ' + str(epoch) + ' fold ' + str(args.fold_idx) + ' acc ' + str(final_acc) + '%\n')

        write_acc.close()

