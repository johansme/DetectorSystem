class NetworkConfig:

    def read_config_from_file(self, filename):  # Should probably change representation to dictionary
        try:
            with open(filename, 'r') as f:
                self.learning_rate = float(f.readline().split()[1])
                self.bidirectional = (f.readline().split()[1] == 'True')
                self.dropout_keep_prob = float(f.readline().split()[1])
                self.l2_coeff = float(f.readline().split()[1])
                self.batch_size = int(f.readline().split()[1])
                num_conv_filters = int(f.readline())
                self.conv_filters_per_layer = []
                for i in range(num_conv_filters):
                    self.conv_filters_per_layer.append(int(f.readline()))
                self.conv_filter_lengths = []
                for i in range(num_conv_filters):
                    self.conv_filter_lengths.append(int(f.readline()))
                self.num_char_lstm_layers = int(f.readline().split()[1])
                self.char_lstm_layer_size = int(f.readline().split()[1])
                self.num_word_lstm_layers = int(f.readline().split()[1])
                self.word_lstm_layer_size = int(f.readline().split()[1])
                num_dense_layers = int(f.readline())
                self.dense_layer_sizes = []
                for i in range(num_dense_layers):
                    self.dense_layer_sizes.append(int(f.readline()))
                self.patience = int(f.readline().split()[1])
        except IOError:
            print('No such config file: {}'.format(filename))

    def save_config_to_file(self, filename):
        try:
            with open(filename, 'w') as f:
                f.write('learning_rate: {}\n'.format(self.learning_rate))
                f.write('bidirectional: {}\n'.format(self.bidirectional))
                f.write('dropout_keep_prob: {}\n'.format(self.dropout_keep_prob))
                f.write('l2_coeff: {}\n'.format(self.l2_coeff))
                f.write('batch_size: {}\n'.format(self.batch_size))
                f.write('{}\n'.format(len(self.conv_filter_lengths)))
                for filters in self.conv_filters_per_layer:
                    f.write('{}\n'.format(filters))
                for length in self.conv_filter_lengths:
                    f.write('{}\n'.format(length))
                f.write('num_char_lstm_layers: {}\n'.format(self.num_char_lstm_layers))
                f.write('char_lstm_layer_size: {}\n'.format(self.char_lstm_layer_size))
                f.write('num_word_lstm_layers: {}\n'.format(self.num_word_lstm_layers))
                f.write('word_lstm_layer_size: {}\n'.format(self.word_lstm_layer_size))
                f.write('{}\n'.format(len(self.dense_layer_sizes)))
                for layer in self.dense_layer_sizes:
                    f.write('{}\n'.format(layer))
                f.write('patience: {}'.format(self.patience))
        except IOError:
            with open(filename, 'x') as f:
                f.write('learning_rate: {}\n'.format(self.learning_rate))
                f.write('bidirectional: {}\n'.format(self.bidirectional))
                f.write('dropout_keep_prob: {}\n'.format(self.dropout_keep_prob))
                f.write('l2_coeff: {}\n'.format(self.l2_coeff))
                f.write('batch_size: {}\n'.format(self.batch_size))
                f.write('{}\n'.format(len(self.conv_filter_lengths)))
                for filters in self.conv_filters_per_layer:
                    f.write('{}\n'.format(filters))
                for length in self.conv_filter_lengths:
                    f.write('{}\n'.format(length))
                f.write('num_char_lstm_layers: {}\n'.format(self.num_char_lstm_layers))
                f.write('char_lstm_layer_size: {}\n'.format(self.char_lstm_layer_size))
                f.write('num_word_lstm_layers: {}\n'.format(self.num_word_lstm_layers))
                f.write('word_lstm_layer_size: {}\n'.format(self.word_lstm_layer_size))
                f.write('{}\n'.format(len(self.dense_layer_sizes)))
                for layer in self.dense_layer_sizes:
                    f.write('{}\n'.format(layer))
                f.write('patience: {}'.format(self.patience))
