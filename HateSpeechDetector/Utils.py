import numpy as np
import matplotlib.pyplot as plt
from random import shuffle


def read_tweets_from_folds(fold_num):
    data = []
    with open('../Data/Folds/Fold{}.txt'.format(fold_num), 'r', encoding='utf-8') as f:
        tweet = ''
        for line in f:
            line = line.strip()
            if line.startswith(':') and line[1:].isdigit():
                data.append([tweet.strip(), int(line[1:])])
                tweet = ''
            else:
                tweet += line + '\n'
    return data


def create_batches(data, mini_batch_size):
    shuffle(data)
    batches = [data[i:i + mini_batch_size] for i in range(0, len(data), mini_batch_size)]
    return batches


def separate_data_from_labels(batch):
    data = [sample[0] for sample in batch]
    labels = [sample[1] for sample in batch]
    return data, labels


def batch_to_3d_array(batch):
    max_el = max(batch, key=len)
    max_len = len(max_el)
    data_dim = len(list(max_el)[0])
    fin_batch = np.array([sample + [[0] * data_dim for _ in range(max_len - len(sample))] for sample in batch])
    return fin_batch


def plot_training_history(error_hist, validation_hist=(), xtitle="Epoch", ytitle="Error",
                          title="History", fig=True):
    plt.ion()
    if fig:
        plt.figure()
    if len(error_hist) > 0:
        simple_plot([p[1] for p in error_hist], [p[0] for p in error_hist],
                    xtitle=xtitle, ytitle=ytitle, title=title)
    if len(validation_hist) > 0:
        simple_plot([p[1] for p in validation_hist], [p[0] for p in validation_hist],
                    xtitle=xtitle, ytitle=ytitle, title=title)
    plt.ioff()
    plt.show(block=False)


def simple_plot(yvals, xvals=None, xtitle='X', ytitle='Y', title='Y = F(X)'):
    xvals = xvals if xvals is not None else list(range(len(yvals)))
    plt.plot(xvals, yvals)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.title(title)
    plt.draw()


def int_to_one_hot(index, depth, on_val=1, off_val=0):
    if index < depth:
        v = [off_val] * depth
        v[index] = on_val
        return v
    else:
        raise ValueError('Tried to create one-hot vector with value{0}, but of size {1}'.format(index, depth))
