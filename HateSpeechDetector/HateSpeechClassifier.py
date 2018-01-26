import tensorflow as tf
import numpy as np


flags = tf.flags

flags.DEFINE_boolean('bidirectional', False, 'Determines whether the LSTM is bidirectional or unidirectional')
flags.DEFINE_boolean('training', True, 'Tells if training is to take place')

FLAGS = flags.FLAGS


class ClassifierNetwork:

    def __init__(self, batch_size, n_gram_len, num_filters, num_char_lstm_layers, word_lstm_layer_size,
                 num_word_lstm_layers, dense_layer_size, learning_rate=.01, bidirectional=False, dropout_keep_prob=1.0):
        self.char_dim = 30
        self.embedding_dim = 300
        self.output_size = 3
        self.batch_size = batch_size
        self.n_gram_length = n_gram_len
        self.filters_per_layer = num_filters
        self.num_con_layers = len(self.filters_per_layer)
        self.num_char_lstm_layers = num_char_lstm_layers

        self.word_lstm_layer_size = word_lstm_layer_size
        self.num_word_lstm_layers = num_word_lstm_layers

        self.dense_layer_size = dense_layer_size

        self.bidirectional = bidirectional
        self.dropout_keep_prob = dropout_keep_prob
        self.learning_rate = learning_rate

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
        char_output, char_lstm_state = self.setup_lstm(conv_output, self.num_char_lstm_layers, tf.shape(conv_output)[2])
        relevant_char_output = last_relevant_from_lstm(char_output, seq_len(char_output))

        word_output, lstm_state = self.setup_lstm(self.word_input, self.num_word_lstm_layers, self.word_lstm_layer_size)
        relevant_word_output = last_relevant_from_lstm(word_output, seq_len(word_output))

        self.fc_input = tf.concat([relevant_char_output, relevant_word_output], 1, name='fully_connected_input')

        self.logits = self.setup_dense_layers()
        self.predictions = {'classes': tf.argmax(self.logits, axis=1),
                            'probabilities': tf.nn.softmax(self.logits, name='softmaxed_output')}

    def setup_training(self):
        self.loss = tf.losses.softmax_cross_entropy(self.target, self.logits)
        self.optimiser = tf.train.AdamOptimizer(learning_rate=self.learning_rate)  # TODO Fill in hyperparameters
        self.training_op = self.optimiser.minimize(self.loss, global_step=tf.train.get_global_step(), name='train_op')

    def generate_cnn_layer(self, layer_input, num_filters, kernel_len, name='conv'):
        with tf.variable_scope(name, reuse=True):
            con = tf.layers.conv1d(layer_input, num_filters, kernel_len, activation=tf.nn.relu, name=name)
        return con

    def setup_cnn(self):
        self.convolutions = []
        con = self.char_input
        for i in range(len(self.filters_per_layer)):
            con = self.generate_cnn_layer(con, self.filters_per_layer[i], self.n_gram_length, 'conv{}'.format(i+1))
            self.convolutions.append(con)
        return con

    def get_lstm_cell(self, layer_size):
        if FLAGS.bidirectional:
            cell = tf.contrib.rnn.BidirectionalGridLSTMCell(layer_size, activation=tf.nn.relu)
        else:
            cell = tf.contrib.rnn.GridLSTMCell(layer_size, activation=tf.nn.relu)
        if FLAGS.training and self.dropout_keep_prob < 1.0:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob)
        return cell

    def setup_lstm(self, lstm_input, n_layers, layer_size):
        cells = tf.contrib.rnn.MultiRNNCell(self.get_lstm_cell(layer_size) for _ in range(n_layers))

        with tf.variable_scope('lstm'):
            output, state = tf.nn.dynamic_rnn(
                cells,
                lstm_input,
                dtype=tf.float32,
                sequence_length=seq_len(lstm_input))
        return output, state

    def setup_dense_layers(self):
        dense = tf.layers.dense(self.fc_input, self.dense_layer_size, activation=tf.nn.relu)

        output = tf.layers.dense(dense, self.output_size, name='unmodified_output')
        return output

    def initialise_session(self):
        init_op = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init_op)
        return sess


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
