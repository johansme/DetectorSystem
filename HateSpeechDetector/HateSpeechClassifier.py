import numpy as np
from random import shuffle
import tensorflow as tf
import Utils


class ClassifierNetwork:

    def __init__(self, config, validation_freq):
        self.config = config

        self.bidirectional = self.config.bidirectional
        self.dropout_keep_prob = self.config.dropout_keep_prob
        self.learning_rate = self.config.learning_rate
        self.l2_coeff = self.config.l2_coeff

        self.char_dim = 31
        self.embedding_dim = 200 if self.config.use_glove else 300
        self.output_size = 3
        self.batch_size = self.config.batch_size
        self.filter_lengths = self.config.conv_filter_lengths
        self.filters_per_layer = self.config.conv_filters_per_layer
        self.num_con_layers = len(self.filters_per_layer)
        self.num_char_lstm_layers = self.config.num_char_lstm_layers
        self.char_lstm_layer_size = self.config.char_lstm_layer_size

        self.word_lstm_layer_size = self.config.word_lstm_layer_size
        self.num_word_lstm_layers = self.config.num_word_lstm_layers

        self.dense_layer_sizes = self.config.dense_layer_sizes

        self.validation_frequency = validation_freq
        self.stopping_patience = self.config.patience

        self.error_history = []
        self.validation_history = []
        self.global_training_step = 0

        self.best_val_error = 100000
        self.steps_since_last_improvement = 0

        self.setup_network()
        self.setup_training()

        self.saver = tf.train.Saver()

        self.current_sess = None

    def setup_network(self):
        tf.reset_default_graph()
        self.char_input = tf.placeholder(tf.float32, shape=[None, None, self.char_dim], name='Char_input')
        self.word_input = tf.placeholder(tf.float32, shape=[None, None, self.embedding_dim], name='Word_input')

        self.dropout_placeholder = tf.placeholder_with_default(1.0, shape=[], name='keep_prob')

        self.target = tf.placeholder(tf.int8, shape=[None, self.output_size], name='Target')

        conv_output = self.setup_cnn()
        char_lengths = seq_len(conv_output)
        char_output, char_lstm_state = self.setup_lstm(conv_output, self.num_char_lstm_layers,
                                                       self.char_lstm_layer_size, 'char_', char_lengths)
        relevant_char_output = last_relevant_from_lstm(char_output, char_lengths) if char_output is not None else None

        word_lengths = seq_len(self.word_input)
        word_output, lstm_state = self.setup_lstm(self.word_input, self.num_word_lstm_layers,
                                                  self.word_lstm_layer_size, 'word_', word_lengths)
        relevant_word_output = last_relevant_from_lstm(word_output, word_lengths) if word_output is not None else None

        if (char_output is not None and word_output is not None):
            self.fc_input = tf.concat([relevant_char_output, relevant_word_output], 1, name='fully_connected_input')
        else:
            if char_output is None and word_output is None:
                raise ValueError('The system requires at least one LSTM layer')
            self.fc_input = relevant_char_output if word_output is None else relevant_word_output

        self.logits = self.setup_dense_layers()
        self.predictions = {'classes': tf.argmax(self.logits, axis=1),
                            'probabilities': tf.nn.softmax(self.logits, name='softmaxed_output')}

    def setup_training(self):
        var = tf.trainable_variables()
        self.l2_loss = self.l2_coeff * tf.add_n([tf.nn.l2_loss(v) for v in var if 'bias' not in v.name])
        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.target, logits=self.logits)
        self.optimiser = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.training_op = self.optimiser.minimize(self.loss + self.l2_loss, global_step=tf.train.get_global_step(),
                                                   name='train_op')

    def generate_cnn_layer(self, layer_input, num_filters, kernel_len, padding='valid', name='conv'):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
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
        cell = tf.contrib.rnn.LSTMCell(layer_size, activation=tf.nn.tanh)
        if self.dropout_keep_prob < 1.0:
            cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.dropout_placeholder,
                   output_keep_prob=self.dropout_placeholder)
        return cell

    def setup_lstm(self, lstm_input, n_layers, layer_size, category, sequence_length):
        if n_layers > 0:
            if self.bidirectional:
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
                        sequence_length=sequence_length)
        else:
            output = None
            state = None
        return output, state

    def setup_dense_layers(self):
        state = self.fc_input
        for i in range(len(self.dense_layer_sizes)):
            state = tf.layers.dense(state, self.dense_layer_sizes[i], activation=tf.nn.relu)
            state = tf.layers.dropout(state, rate=1 - self.dropout_placeholder, training=self.dropout_placeholder < 1.0)
        output = tf.layers.dense(state, self.output_size, name='unmodified_output')
        return output

    def initialise_session(self):
        init_op = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init_op)
        self.current_sess = sess

    def initialise_session_ex_vars(self):
        self.current_sess = tf.Session()

    def do_early_stopping(self, step):
        if step % self.validation_frequency == 0:
            new_error = self.do_validation(self.validation_data, step)
            if new_error < self.best_val_error:
                self.best_val_error = new_error
                self.steps_since_last_improvement = 0
                self.best_val_path = self.saver.save(self.current_sess, 'temp/training_temp')
            elif self.steps_since_last_improvement > self.stopping_patience:
                return True
            else:
                self.steps_since_last_improvement += 1
        return False

    def do_training(self, epochs):
        if self.current_sess is None:
            self.initialise_session()
        sess = self.current_sess
        step = self.global_training_step
        for i in range(epochs):
            step = self.global_training_step + i
            error = 0
            for batch in self.training_data:
                char_input = batch[0]
                word_input = batch[1]
                targets = batch[2]
                feeder = {self.char_input: char_input, self.word_input: word_input, self.target: targets,
                          self.dropout_placeholder: self.dropout_keep_prob}
                _, loss = sess.run([self.training_op, self.loss], feed_dict=feeder)
                error += sum(loss)/len(batch)
            self.error_history.append((step, error/len(self.training_data)))  # len(training_data) = num mini batches
            stop = self.do_early_stopping(step)
            if stop:
                print('Training stopped at step {}'.format(step))
                break
            shuffle(self.training_data)
        self.load_model(self.current_sess, self.best_val_path)
        self.global_training_step = step
        Utils.plot_training_history(self.error_history, self.validation_history)

    def do_validation(self, validation_data, step):
        if self.current_sess is None:
            self.initialise_session()
        error = 0
        for batch in validation_data:
            char_input = batch[0]
            word_input = batch[1]
            targets = batch[2]
            feeder = {self.char_input: char_input, self.word_input: word_input, self.target: targets}
            loss = self.current_sess.run([self.loss], feed_dict=feeder)
            error += sum(loss[0])/len(batch)
        self.validation_history.append((step, error/len(validation_data)))
        return error

    def do_testing(self, test_data):
        if self.current_sess is None:
            self.initialise_session()
        confusion = np.zeros([self.output_size, self.output_size], dtype=int)
        for batch in test_data:
            char_input = batch[0]
            word_input = batch[1]
            targets = batch[2]
            feeder = {self.char_input: char_input, self.word_input: word_input, self.target: targets}
            predictions = self.current_sess.run([self.predictions], feed_dict=feeder)
            classes = predictions[0]['classes']  # run returns a singleton list, hence element 0
            with self.current_sess.as_default():
                targets = tf.argmax(targets, axis=1).eval()
            for i in range(len(targets)):
                confusion[classes[i]][targets[i]] += 1
        print('Confusion matrix of the test data:\n\tTargets:\nNeutral\tSexist\tRacist\n{}'.format(confusion))

    def save_model(self, sess, filename):
        save_path = self.saver.save(sess, 'temp/' + filename)
        print('Model saved to: {}'.format(save_path))

    def load_model(self, sess, save_path):
        self.saver.restore(sess, save_path)

    def set_data(self, training_data, validation_data):
        self.training_data = training_data
        self.validation_data = validation_data


def last_relevant_from_lstm(output, length):
    length = increment_zeros(length)  # Necessary to avoid errors for zero-length cases
    batch_size = tf.shape(output)[0]
    max_len = tf.shape(output)[1]
    out_size = output.get_shape()[2].value
    ind = tf.range(0, batch_size) * max_len + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    return tf.gather(flat, ind)


def seq_len(sequence):
    used = tf.sign(tf.reduce_max(sequence, 2))
    length = tf.reduce_sum(used, 1)
    return tf.cast(length, tf.int32)


def increment_zeros(length_tensor):
    ones = tf.ones_like(length_tensor)
    return tf.where(tf.cast(length_tensor, tf.bool), x=length_tensor, y=ones)
