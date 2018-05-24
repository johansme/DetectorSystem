import InputHandler
import HateSpeechClassifier
import matplotlib.pyplot as plt
import NetworkConfig
from random import shuffle
import Utils


def do_10fold_cross_validation(config_name):
    config = NetworkConfig.NetworkConfig()
    config.read_config_from_file(config_name)
    folds = []
    for i in range(1, 11):
        fold = Utils.read_tweets_from_folds(i)
        fold = Utils.create_batches(fold, config.batch_size)
        folds.append(fold)
    word2vec = InputHandler.generate_word2vec_model(config.use_glove)
    for i, fold in enumerate(folds):
        fold_batches = []
        for batch in fold:
            data, labels = Utils.separate_data_from_labels(batch)
            word_input = InputHandler.preprocess_word_based(data, word2vec)
            word_input = Utils.batch_to_3d_array(word_input)
            char_input = InputHandler.preprocess_character_based(data)
            char_input = Utils.batch_to_3d_array(char_input)
            for j, lab in enumerate(labels):
                labels[j] = Utils.int_to_one_hot(lab, 3)
            fold_batches.append([char_input, word_input, labels])
        folds[i] = fold_batches
    for i in range(10):
        val_fold = folds[i]
        train_folds = folds[:i] + (folds[i+1:] if i < 9 else [])
        run_cross_validation_round(train_folds, val_fold, config)
    plt.show(block=True)


def run_cross_validation_round(training_folds, test_fold, config):
    classifier = HateSpeechClassifier.ClassifierNetwork(config, 3)
    training_data = []
    for fold in training_folds:
        training_data.extend(fold)
    shuffle(training_data)
    classifier.set_data(training_data=training_data, validation_data=test_fold)
    classifier.do_training(100)
    classifier.do_testing(test_data=test_fold)


do_10fold_cross_validation(input('Name of config: '))
