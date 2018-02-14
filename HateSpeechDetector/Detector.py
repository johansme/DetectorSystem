import InputHandler
import HateSpeechClassifier
import Utils
import tensorflow as tf


def test_classifier():
    classifier = HateSpeechClassifier.ClassifierNetwork('TestConfig.txt', 20)
    data = Utils.read_tweets_from_folds(1)
    data = Utils.create_batches(data, 20)
    word2vec = InputHandler.generate_word2vec_model()
    full_data = []
    for batch in data:
        data, labels = Utils.separate_data_from_labels(batch)
        word_input = InputHandler.preprocess_word_based(data, word2vec)
        word_input = Utils.batch_to_3d_array(word_input)
        char_input = InputHandler.preprocess_character_based(data)
        char_input = Utils.batch_to_3d_array(char_input)
        labels = tf.one_hot(labels, 3)
        full_data.append([char_input, word_input, labels])
    word2vec = None
    classifier.set_data(full_data[:-2], full_data[-2])
    classifier.do_training(full_data[:-2], 20)


test_classifier()
