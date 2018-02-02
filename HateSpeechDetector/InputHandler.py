from gensim.models import KeyedVectors
import os
import preprocessor as p
import tensorflow as tf
import numpy as np


alphabet = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9, 'k': 10, 'l': 11, 'm': 12,
            'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19, 'u': 20, 'v': 21, 'w': 22, 'x': 23, 'y': 24,
            'z': 25, ' ': 26, '#': 27, '@': 28}


def generate_word2vec_model():
    file_dir = os.path.dirname(os.path.realpath('__file__'))
    mod_path = os.path.join(file_dir, '../Data/word2vec/GoogleNews-vectors-negative300.bin')
    model = KeyedVectors.load_word2vec_format(mod_path, binary=True)
    return model


def preprocess_character_based(tweets):
    p.set_options(p.OPT.EMOJI)  # Include urls?
    indx = []
    for tweet in tweets:
        clean = p.clean(tweet)
        clean = clean.lower()
        chars = []
        for ch in clean:
            if ch in alphabet:
                chars.append(alphabet[ch])
            else:
                chars.append(29)
        indx.append(chars)
    batch = tf.one_hot(indx, 30)
    return batch


def preprocess_word_based(tweets, vocab_model):
    p.set_options(p.OPT.EMOJI, p.OPT.URL, p.OPT.MENTION, p.OPT.HASHTAG)
    batch = []
    for tweet in tweets:
        # Should I remove hashtags completely, or just remove the symbol?
        clean = p.clean(tweet)
        clean = clean.lower()


def equalise_batch_length(batch):
    max_len = max(batch, key=lambda x: len(x))
    data_dim = len(batch[0][0])
    fin_batch = np.array([sample + [[0]*data_dim for _ in (max_len - len(sample))] for sample in batch])
    return fin_batch
