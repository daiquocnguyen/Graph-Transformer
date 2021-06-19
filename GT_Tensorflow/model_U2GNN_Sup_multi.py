import tensorflow as tf
import universal_transformer_modified

class U2GNN(object):
    def __init__(self, feature_dim_size, hparams_batch_size, ff_hidden_size, seq_length, num_classes, num_self_att_layers, num_U2GNN_layers=1):
        # Placeholders for input, output
        self.input_x = tf.compat.v1.placeholder(tf.int32, [None, seq_length], name="input_x")
        self.graph_pool = tf.compat.v1.sparse_placeholder(tf.float32, [None, None], name="graph_pool")
        self.X_concat = tf.compat.v1.placeholder(tf.float32, [None, feature_dim_size], name="X_concat")
        self.one_hot_labels = tf.compat.v1.placeholder(tf.float32, [None, num_classes], name="one_hot_labels")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        #Inputs for Universal Transformer
        self.input_UT = tf.nn.embedding_lookup(self.X_concat, self.input_x)
        self.input_UT = tf.reshape(self.input_UT, [-1, seq_length, 1, feature_dim_size])

        #Matrix weights in Universal Transformer are shared across each attention layer (timestep), while they are not in Transformer.
        #It's optional to use Transformer Encoder.
        self.hparams = universal_transformer_modified.universal_transformer_small1()
        self.hparams.hidden_size = feature_dim_size
        self.hparams.batch_size = hparams_batch_size * seq_length
        self.hparams.max_length = seq_length
        self.hparams.num_hidden_layers = num_self_att_layers  # Number of attention layers: the number T of timesteps in Universal Transformer
        self.hparams.num_heads = 1  # due to the fact that the feature embedding sizes are various
        self.hparams.filter_size = ff_hidden_size
        self.hparams.use_target_space_embedding = False
        self.hparams.pos = None
        self.hparams.add_position_timing_signal = False
        self.hparams.add_step_timing_signal = False
        self.hparams.add_sru = False
        self.hparams.add_or_concat_timing_signal = None

        # Construct k GNN layers
        self.scores = 0
        for layer in range(num_U2GNN_layers):  # the number k of multiple stacked layers, each stacked layer includes a number of self-attention layers
            # Universal Transformer Encoder
            self.ute = universal_transformer_modified.UniversalTransformerEncoder1(self.hparams, mode=tf.estimator.ModeKeys.TRAIN)
            self.output_UT = self.ute({"inputs": self.input_UT, "targets": 0, "target_space_id": 0})[0]
            self.output_UT = tf.squeeze(self.output_UT, axis=2)
            #
            self.output_target_node = tf.split(self.output_UT, num_or_size_splits=seq_length, axis=1)[0]
            self.output_target_node = tf.squeeze(self.output_target_node, axis=1)
            #input for next GNN hidden layer
            self.input_UT = tf.nn.embedding_lookup(self.output_target_node, self.input_x)
            self.input_UT = tf.reshape(self.input_UT, [-1, seq_length, 1, feature_dim_size])
            # graph pooling
            self.graph_embeddings = tf.compat.v1.sparse_tensor_dense_matmul(self.graph_pool, self.output_target_node)
            self.graph_embeddings = tf.nn.dropout(self.graph_embeddings, keep_prob=self.dropout_keep_prob)

            # Concatenate graph representations from all GNN layers
            with tf.variable_scope("layer_%d" % layer):
                W = tf.compat.v1.get_variable(shape=[feature_dim_size, num_classes],
                                              initializer=tf.contrib.layers.xavier_initializer(),
                                              name="W_layer_%d" % layer)
                b = tf.Variable(tf.zeros([num_classes]))
                self.scores += tf.compat.v1.nn.xw_plus_b(self.graph_embeddings, W, b)

        # Final predictions
        self.predictions = tf.argmax(self.scores, 1, name="predictions")
        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=label_smoothing(self.one_hot_labels))
            self.total_loss = tf.reduce_mean(losses)

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.one_hot_labels, 1))
            self.accuracy = tf.reduce_sum(tf.cast(correct_predictions, "float"), name="accuracy")

        self.saver = tf.compat.v1.train.Saver(tf.global_variables(), max_to_keep=500)
        tf.logging.info('Seting up the main structure')

def label_smoothing(inputs, epsilon=0.1):
    V = inputs.get_shape().as_list()[-1]  # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / V)
