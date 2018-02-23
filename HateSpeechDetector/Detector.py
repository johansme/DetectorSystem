import InputHandler
import HateSpeechClassifier
import Utils


def test_classifier():
    classifier = HateSpeechClassifier.ClassifierNetwork('TestConfig.txt', 20)
    data = Utils.read_tweets_from_folds(1)
    data = Utils.create_batches(data, classifier.batch_size)
    word2vec = InputHandler.generate_word2vec_model()
    full_data = []
    for batch in data:
        data, labels = Utils.separate_data_from_labels(batch)
        word_input = InputHandler.preprocess_word_based(data, word2vec)
        word_input = Utils.batch_to_3d_array(word_input)
        char_input = InputHandler.preprocess_character_based(data)
        char_input = Utils.batch_to_3d_array(char_input)
        for i, lab in enumerate(labels):
            labels[i] = Utils.int_to_one_hot(lab, 3)
        full_data.append([char_input, word_input, labels])
    word2vec = None
    classifier.set_data(training_data=full_data[:-2], validation_data=full_data[-2:])
    classifier.do_training(21)


def do_10fold_cross_validation(config_name):
    classifier = HateSpeechClassifier.ClassifierNetwork(config_name, 20)
    folds = []
    for i in range(1, 11):
        fold = Utils.read_tweets_from_folds(i)
        fold = Utils.create_batches(fold, classifier.batch_size)
        folds.append(fold)
    word2vec = InputHandler.generate_word2vec_model()
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
        run_cross_validation_round(train_folds, val_fold, classifier)


def run_cross_validation_round(training_folds, validation_fold, classifier):
    training_data = []
    for fold in training_folds:
        training_data.extend(fold)
    classifier.set_data(training_data=training_data, validation_data=validation_fold)
    classifier.do_training(1000)
    classifier.do_testing(test_data=validation_fold)


do_10fold_cross_validation('TestConfig.txt')
