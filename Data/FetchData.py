# -*- coding: utf-8 -*-
import twitter


def read_900_tweets(start_num):
    api = twitter.Api()
    cases = []
    deleted = 0
    with open('NAACL_SRW_2016.csv', 'r') as f:
        for i in range(start_num):
            f.readline()
        for i in range(900):
            line = f.readline().strip()
            if line != '':
                tid, label = line.split(';')
                try:
                    status = api.GetStatus(tid, trim_user=True, include_my_retweet=False, include_entities=False)
                    obj = status.AsDict()
                    text = obj.get('text')
                    if text is not None and text != '':
                        text = text.replace('&amp;', '&')
                        cases.append((text, label))
                except twitter.error.TwitterError:
                    deleted += 1
    return cases, deleted


def fetch_data():
    with open('StartLine.txt', 'r') as f:
        start_num = int(f.readline())
    if start_num < 16907:
        new_cases, deleted = read_900_tweets(start_num)

        for case in new_cases:
            if case[1] == 'racism':
                file_name = 'RacismSamples.txt'
            elif case[1] == 'sexism':
                file_name = 'SexismSamples.txt'
            else:
                file_name = 'NeutralSamples.txt'
            with open(file_name, 'a', encoding='utf-8') as f:
                f.write(case[0] + '\n')
                f.write(';\n')

        with open('StartLine.txt', 'w') as f:
            f.write(str(start_num + 900))
        with open('DeletedPerBatch.txt', 'a') as f:
            f.write(str(deleted) + '\n')
        print('Batch fetched, {0} tweets deleted'.format(deleted))
    else:
        print('All tweets fetched')


fetch_data()
