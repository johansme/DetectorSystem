from gensim.models import Word2Vec
import os
import preprocessor as p


def generate_word2vec_model():
    file_dir = os.path.dirname(os.path.realpath('__file__'))
    mod_path = os.path.join(file_dir, '../Data/word2vec/GoogleNews-vectors-negative300.bin')
    model = Word2Vec.load_word2vec_format(mod_path, binary=True)
    return model


def preprocess_character_based(tweets):
    p.set_options(p.OPT.EMOJI)
    for tweet in tweets:
        pass


def preprocess_word_based(tweets):
    p.set_options(p.OPT.EMOJI, p.OPT.URL, p.OPT.MENTION, p.OPT.HASHTAG)
    for tweet in tweets:
        # Should I remove hashtags completely, or just remove the symbol?
        res = p.clean(tweet)
