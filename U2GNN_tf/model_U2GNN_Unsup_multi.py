import tensorflow as tf
import universal_transformer_modified

class U2GNN(object):
    def __init__(self, vocab_size, feature_dim_size, hparams_batch_size, ff_hidden_size, num_sampled, seq_length, num_self_att_layers, num_U2GNN_layers=1):
        # Placeholders for input, output
        self.X_concat = tf.compat.v1.placeholder(tf.float32, [None, feature_dim_size], name="X_concat")
        self.input_x = tf.compat.v1.placeholder(tf.int32, [None, seq_length], name="input_x")
        self.input_y = tf.compat.v1.placeholder(tf.int32, [None, 1], name="input_y")
        self.dropout_keep_prob = tf.compat.v1.placeholder(tf.float32, name="dropout_keep_prob")

        # Inputs for Universal Transformer
        self.input_UT = tf.nn.embedding_lookup(self.X_concat, self.input_x)
        self.input_UT = tf.reshape(self.input_UT, [-1, seq_length, 1, feature_dim_size])

        # Matrix weights in Universal Transformer are shared across each attention layer (timestep), while they are not in Transformer.
        # It's optional to use Transformer Encoder.
        self.hparams = universal_transformer_modified.universal_transformer_small1()
        self.hparams.hidden_size = feature_dim_size
        self.hparams.batch_size = hparams_batch_size * seq_length
        self.hparams.max_length = seq_length
        self.hparams.num_hidden_layers = num_self_att_layers # Number of attention layers: the number T of timesteps in Universal Transformer
        self.hparams.num_heads = 1 #due to the fact that the feature embedding sizes are various
        self.hparams.filter_size = ff_hidden_size
        self.hparams.use_target_space_embedding = False
        self.hparams.pos = None
        self.hparams.add_position_timing_signal = False
        self.hparams.add_step_timing_signal = False
        self.hparams.add_sru = False
        self.hparams.add_or_concat_timing_signal = None

        self.output_vectors = []
        #Construct k GNN layers
        for layer in range(num_U2GNN_layers):  # The number k of multiple stacked layers, each stacked layer includes a number of self-attention layers
            # Universal Transformer Encoder
            self.ute = universal_transformer_modified.UniversalTransformerEncoder1(self.hparams, mode=tf.estimator.ModeKeys.TRAIN)
            self.output_UT = self.ute({"inputs": self.input_UT, "targets": 0, "target_space_id": 0})[0]
            self.output_UT = tf.squeeze(self.output_UT, axis=2)
            #
            self.output_target_node = tf.split(self.output_UT, num_or_size_splits=seq_length, axis=1)[0]
            self.output_target_node = tf.squeeze(self.output_target_node, axis=1)
            #
            self.output_vectors.append(self.output_target_node)
            #input for next GNN hidden layer
            self.input_UT = tf.nn.embedding_lookup(self.output_target_node, self.input_x)
            self.input_UT = tf.reshape(self.input_UT, [-1, seq_length, 1, feature_dim_size])

        self.output_vectors = tf.concat(self.output_vectors, axis=1)
        self.output_vectors = tf.nn.dropout(self.output_vectors, keep_prob=self.dropout_keep_prob)

        with tf.name_scope("embedding"):
            self.embedding_matrix = tf.compat.v1.get_variable(
                    "W", shape=[vocab_size, feature_dim_size*num_U2GNN_layers], initializer=tf.contrib.layers.xavier_initializer())
            self.softmax_biases = tf.Variable(tf.zeros([vocab_size]))

        self.total_loss = tf.reduce_mean(
            tf.nn.sampled_softmax_loss(weights=self.embedding_matrix, biases=self.softmax_biases, inputs=self.output_vectors,
                                       labels=self.input_y, num_sampled=num_sampled, num_classes=vocab_size))

        self.saver = tf.compat.v1.train.Saver(tf.global_variables(), max_to_keep=500)
        tf.logging.info('Seting up the main structure')

