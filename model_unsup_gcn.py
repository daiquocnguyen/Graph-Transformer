import tensorflow as tf
from gcn_layer import *

class GCN_graph_cls(object):
    def __init__(self, feature_dim_size, hidden_size, num_GNN_layers, num_sampled, vocab_size):
        # Placeholders for input, output
        self.Adj_block = tf.compat.v1.sparse_placeholder(tf.float32, [None, None], name="Adj_block")
        self.X_concat = tf.compat.v1.sparse_placeholder(tf.float32, [None, feature_dim_size], name="X_concat")
        self.num_features_nonzero = tf.compat.v1.placeholder(tf.int32, name="num_features_nonzero")
        self.dropout = tf.compat.v1.placeholder(tf.float32, name="dropout")
        self.input_y = tf.compat.v1.placeholder(tf.int32, [None, 1], name="input_y")

        self.placeholders = {
            'adj': self.Adj_block,
            'dropout': self.dropout,
            'num_features_nonzero': self.num_features_nonzero
        }

        self.input = self.X_concat   # set hidden_size = feature_dim_size if not tuning sizes of hidden stacked layers
        in_hidden_size = feature_dim_size
        self.output_vectors = []
        #Construct k GNN layers
        for idx_layer in range(num_GNN_layers):
            sparse_inputs = False
            if idx_layer == 0:
                sparse_inputs = True
            gcn_gnn = GraphConvolution(input_dim=in_hidden_size,
                                                  output_dim=hidden_size,
                                                  placeholders=self.placeholders,
                                                  act=tf.nn.relu,
                                                  dropout=True,
                                                  sparse_inputs=sparse_inputs)

            in_hidden_size = hidden_size
            # run --> output --> input for next layer
            self.input = gcn_gnn(self.input)
            #
            self.output_vectors.append(self.input)

        self.output_vectors = tf.concat(self.output_vectors, axis=1)
        self.output_vectors = tf.nn.dropout(self.output_vectors, 1-self.dropout)

        with tf.name_scope("embedding"):
            self.embedding_matrix = glorot([vocab_size, hidden_size*num_GNN_layers], name='node_embeddings')
            self.softmax_biases = tf.Variable(tf.zeros([vocab_size]))

        self.total_loss = tf.reduce_mean(
            tf.nn.sampled_softmax_loss(weights=self.embedding_matrix, biases=self.softmax_biases,
                                       inputs=self.output_vectors, labels=self.input_y, num_sampled=num_sampled, num_classes=vocab_size))

        self.saver = tf.compat.v1.train.Saver(tf.global_variables(), max_to_keep=500)
        tf.logging.info('Seting up the main structure')