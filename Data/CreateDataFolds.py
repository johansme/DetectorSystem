from random import shuffle


def create_balanced_folds(num_folds):
    neutral_samples = read_tweets_from_file('NeutralSamples.txt', 0)
    sexist_samples = read_tweets_from_file('SexismSamples.txt', 1)
    racist_samples = read_tweets_from_file('RacismSamples.txt', 2)
    shuffle(sexist_samples)
    shuffle(racist_samples)
    shuffle(neutral_samples)
    sexist_samples = partition(sexist_samples, num_folds)
    racist_samples = partition(racist_samples, num_folds)
    racist_samples.reverse()  # To reduce accumulated impact of the first elements being the longest
    neutral_samples = partition(neutral_samples, num_folds)
    for i in range(num_folds):
        fold = sexist_samples[i] + racist_samples[i] + neutral_samples[i]
        shuffle(fold)
        write_tweets_to_file('Folds/Fold'+str(i+1)+'.txt', fold)


def partition(lst, num_partitions):
    return [lst[i::num_partitions] for i in range(num_partitions)]


def read_tweets_from_file(filename, label):
    tweets = []
    try:
        with open(filename, 'r', encoding='utf-8', newline='\n') as f:
            tweet = ''
            for line in f:
                line = line.strip()
                if line == ';':
                    tweets.append([tweet.strip(), label])
                    tweet = ''
                else:
                    tweet += line + '\n'
    except IOError:
        print('Something went wrong reading from file {}'.format(filename))
    return tweets


def write_tweets_to_file(filename, tweets):
    with open(filename, 'w', encoding='utf-8') as f:
        try:
            for tweet in tweets:
                f.write(tweet[0])
                f.write('\n:{}\n'.format(tweet[1]))
        except IOError:
            print('Something went wrong in writing to file {}'.format(filename))


create_balanced_folds(10)
