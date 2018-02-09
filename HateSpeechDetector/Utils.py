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
    batches = [data[i:i+mini_batch_size] for i in range(0, len(data), mini_batch_size)]
    return batches
