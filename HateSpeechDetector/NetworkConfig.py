class NetworkConfig:

    def read_config_from_file(self, filename):
        try:
            with open(filename, 'r') as f:
                self.learning_rate = int(f.readline().split()[1])
                self.batch_size = int(f.readline().split()[1])
                num_conv_filters = int(f.readline())
                self.conv_filters_per_layer = []
                for i in range(num_conv_filters):
                    self.conv_filters_per_layer.append(int(f.readline()))
                self.conv_filter_lengths = []
                for i in range(num_conv_filters):
                    self.conv_filter_lengths.append(int(f.readline()))
                self.num_char_lstm_layers = int(f.readline().split()[1])
        except IOError:
            print('No such config file: {}'.format(filename))

    def save_config_to_file(self, filename):
        try:
            with open(filename, 'w') as f:
                f.write('learning_rate: {}\n'.format(self.learning_rate))
                f.write('batch_size: {}\n'.format(self.batch_size))
                num_conv_filters = len(self.conv_filter_lengths)
                f.write('{}\n'.format(num_conv_filters))
                for i in range(num_conv_filters):
                    f.write('{}\n'.format(self.conv_filters_per_layer[i]))
                for i in range(num_conv_filters):
                    f.write('{}\n'.format(self.conv_filter_lengths[i]))
                f.write('num_char_lstm_layers: {}\n'.format(self.num_char_lstm_layers))
        except IOError:
            with open(filename, 'x') as f:
                f.write('learning_rate: {}\n'.format(self.learning_rate))
                f.write('batch_size: {}\n'.format(self.batch_size))