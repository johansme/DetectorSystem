import tensorflow as tf
import numpy as np
import NetworkConfig


flags = tf.flags

flags.DEFINE_boolean('bidirectional', False, 'Determines whether the LSTM is bidirectional or unidirectional')
flags.DEFINE_boolean('training', True, 'Tells if training is to take place')

FLAGS = flags.FLAGS

# TODO Add regularisation of nodes, try LR instead of dense layers (binary)?
class ClassifierNetwork:

    def __init__(self, config_name, num_char_lstm_layers, word_lstm_layer_size, num_word_lstm_layers,
                 dense_layer_sizes, learning_rate=.01, bidirectional=False, dropout_keep_prob=1.0):
        self.config = NetworkConfig.NetworkConfig()
        self.config.read_config_from_file(config_name)
        self.char_dim = 30
        self.embedding_dim = 300
        self.output_size = 3
        self.batch_size = self.config.batch_size
        self.filter_lengths = self.config.conv_filter_lengths
        self.filters_per_layer = self.config.conv_filters_per_layer
        self.num_con_layers = len(self.filters_per_layer)
        self.num_char_lstm_layers = num_char_lstm_layers

        self.word_lstm_layer_size = word_lstm_layer_size
        self.num_word_lstm_layers = num_word_lstm_layers

        self.dense_layer_sizes = dense_layer_sizes

        self.bidirectional = bidirectional
        self.dropout_keep_prob = dropout_keep_prob
        self.learning_rate = learning_rate

        self.saver = tf.train.Saver()

        self.setup_network()
        self.setup_training()

        self.current_sess = self.initialise_session()

    def setup_network(self):
        tf.reset_default_graph()
        self.char_input = tf.placeholder(tf.int8, shape=[self.batch_size, None, self.char_dim], name='Char_input')
        self.word_input = tf.placeholder(tf.float32, shape=[self.batch_size, None, self.embedding_dim],
                                         name='Word_input')

        self.target = tf.placeholder(tf.int8, shape=[None, self.output_size], name='Target')

        # TODO fix padding to equalise length of sentences in mini_batch
        conv_output = self.setup_cnn()
        char_lengths = seq_len(conv_output)
        char_output, char_lstm_state = self.setup_lstm(conv_output, self.num_char_lstm_layers,
                                                       tf.shape(conv_output)[2], 'char_', char_lengths)
        relevant_char_output = last_relevant_from_lstm(char_output, char_lengths)

        word_lengths = seq_len(self.word_input)
        word_output, lstm_state = self.setup_lstm(self.word_input, self.num_word_lstm_layers,
                                                  self.word_lstm_layer_size, 'word_', word_lengths)
        relevant_word_output = last_relevant_from_lstm(word_output, word_lengths)

        self.fc_input = tf.concat([relevant_char_output, relevant_word_output], 1, name='fully_connected_input')

        self.logits = self.setup_dense_layers()
        self.predictions = {'classes': tf.argmax(self.logits, axis=1),
                            'probabilities': tf.nn.softmax(self.logits, name='softmaxed_output')}

    def setup_training(self):
        self.loss = tf.losses.softmax_cross_entropy(self.target, self.logits)
        self.optimiser = tf.train.AdamOptimizer(learning_rate=self.learning_rate)  # TODO Fill in hyperparameters
        self.training_op = self.optimiser.minimize(self.loss, global_step=tf.train.get_global_step(), name='train_op')

    def generate_cnn_layer(self, layer_input, num_filters, kernel_len, padding='valid', name='conv'):
        with tf.variable_scope(name, reuse=True):
            con = tf.layers.conv1d(layer_input, num_filters, kernel_len, padding=padding,
                                   activation=tf.nn.relu, name=name)
        return con

    def setup_cnn(self):
        self.convolutions = []
        con = self.char_input
        for i in range(len(self.filters_per_layer)):
            if i == 0:
                padding = 'same'
            else:
                padding = 'valid'
            con = self.generate_cnn_layer(con, self.filters_per_layer[i], self.filter_lengths[i],
                                          padding=padding, name='conv{}'.format(i+1))
            self.convolutions.append(con)
        return con

    def get_lstm_cell(self, layer_size):
        cell = tf.contrib.rnn.LSTMCell(layer_size, activation=tf.nn.relu)
        if FLAGS.training and self.dropout_keep_prob < 1.0:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob)
        return cell

    def setup_lstm(self, lstm_input, n_layers, layer_size, category, sequence_length):
        if FLAGS.bidirectional:
            with tf.variable_scope(category + 'lstm'):
                output, fw_state, bw_state = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                    [self.get_lstm_cell(layer_size) for _ in range(n_layers)],
                    [self.get_lstm_cell(layer_size) for _ in range(n_layers)],
                    lstm_input,
                    dtype=tf.float32,
                    sequence_length=sequence_length
                )
                state = (fw_state, bw_state)
        else:
            cells = tf.contrib.rnn.MultiRNNCell([self.get_lstm_cell(layer_size) for _ in range(n_layers)])
            with tf.variable_scope(category + 'lstm'):
                output, state = tf.nn.dynamic_rnn(
                    cells,
                    lstm_input,
                    dtype=tf.float32,
                    sequence_length=seq_len(lstm_input))
        return output, state

    def setup_dense_layers(self):
        state = self.fc_input
        for i in range(len(self.dense_layer_sizes)):
            state = tf.layers.dense(state, self.dense_layer_sizes[i], activation=tf.nn.relu)
        output = tf.layers.dense(state, self.output_size, name='unmodified_output')  # TODO Should there be activation?
        return output

    def initialise_session(self):
        init_op = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init_op)
        return sess

    def initialise_session_ex_vars(self):
        return tf.Session()

    def save_model(self, sess, filename):
        save_path = self.saver.save(sess, 'temp/' + filename + '.ckpt')
        print('Model saved to: {}'.format(save_path))

    def load_model(self, sess, save_path):
        self.saver.restore(sess, save_path)


def last_relevant_from_lstm(output, length):
    batch_size = tf.shape(output)[0]
    max_len = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    ind = tf.range(0, batch_size) * max_len + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    return tf.gather(flat, ind)


def seq_len(sequence):
    used = tf.sign(tf.reduce_max(sequence, 2))
    length = tf.reduce_sum(used, 1)
    return tf.cast(length, tf.int32)
