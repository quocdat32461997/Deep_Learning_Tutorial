#  hw1.py
# Author: Dat Quoc Ngo
# NET-ID: dqn170000
# Course: CS6364.002

import os
import json
import datetime
from collections import defaultdict
from pattern.en import sentiment, positive

def task1():
    # task 1 - hello world
    print('Hello World')
    return None

if __name__ == '__main__':
    task1()

def task2():
    # task 2 - define object
    items = [1, 2, 3, 4, 5]
    print(items)
    return None

def task3():
    # task 3- file reading
    with open('task3.data', 'r') as file:
        items = file.read()
        items = items.split() # split by space by default
    # split items into 2 5-item list
    items1 = items[:5]
    items2 = items[5:]
    print('Items 1: {}'.format(items1))
    print('Items 2: {}'.format(items2))

def task4():
    # task 4 - programming w/ data structures
    data = dict()
    data['school'] = 'UAlbany'
    data['address'] = '1400 Washington Ave, Albany, NY 12222'
    data['phone'] = '(518) 442-3300'
    data = {'school': 'UAlbany', 'address': '1400 Washington Ave, Albany, NY 12222', 'phone': '(518) 442-3300'}

    # print dict
    for x, y in  data .items():
        print('{}: {}'.format(x, y))

def task5():
    # task 5 - serialize data
    data = {'school': 'UAlbany', 'address': '1400 Washington Ave, Albany, NY 12222', 'phone': '(518) 442-3300'}

    # dump data
    x = json.dumps(data)

    # load data
    data = json.loads(x)
    print(data)

def task6():
    # task 6- serialize any data type into json
    items = [1,2,3,4,5]
    data = {'school': 'UAlbany', 'address': '1400 Washington Ave, Albany, NY 12222', 'phone': '(518) 442-3300'}

    # dump data and items into task6.data
    with open('task6.data', 'w') as file:
        json.dump({'items': items, 'data': data}, file)

    # load data and print
    with open('task6.data') as file:
        data = json.load(file)

    # print json file
    for x, y in data.items():
        print('{}: {}'.format(x, y))

def task7():
    # task 7 - data preprocessing

    # read tweets
    with open('CrimeReport.txt') as file:
        tweets = [json.loads(tweet) for tweet in file.readlines()]

    # print id for every tweet
    for tweet in tweets:
        # convert string to dict object
        print('{}: {}'.format(tweet['id'], tweet['text']))

def task8():
    # task 8 - tweets filtering

    # read tweets
    with open('CrimeReport.txt') as file:
        tweets = [json.loads(tweet) for tweet in file.readlines()]

    # convert timestamp from str to datetime object
    # and sort
    sorted_tweets = sorted(tweets, key = lambda item:
                           datetime.datetime.strptime(item['created_at'], '%a %b %d %H:%M:%S +0000 %Y'))

    # save 10 most recent tweets
    with open('task8.data', 'w') as file:
        # get 10 most recent tweets
        for tweet in sorted_tweets[-10:]:
            file.write(json.dumps(tweet) + '\n')

def task9():
    # task 9 - file operations

    # read tweets
    tweets = defaultdict(datetime.date)
    with open('CrimeReport.txt') as file:
        for tweet in file.readlines():
            tweet = json.loads(tweet)

            # convert time in str too datetime object
            time_stamp = datetime.datetime.strptime(tweet['created_at'], '%a %b %d %H:%M:%S +0000 %Y')
            # group to previous hour
            time_stamp = time_stamp.replace(microsecond=0, second=0, minute=0)

            # group tweets to hours
            if time_stamp not in tweets:
                tweets[time_stamp] = [tweet]
            else:
                tweets[time_stamp].append(tweet)

    # save file according to time stamp
    if not os.path.exists('task9-output'):
        os.mkdir('task9-output')
    for ts, tws in tweets.items():
        with open('task9-output/{}-{}-{}-{}.txt'.format(ts.month, ts.day, ts.year, ts.hour), 'w') as file:
            for tw in tws:
                file.write(json.dumps(tw) + '\n')

def task10():
    # read text
    with open('CrimeReport.txt') as file:
        tweets = [json.loads(tweet) for tweet in file.readlines()]

    # open writing to positive and negative sents
    pos_tweets = open('positive-sentiment-tweets.txt', 'w')
    neg_tweets = open('negative-sentiment-tweets.txt', 'w')
    # compute polarity and determine if tweet is positive or not
    for tweet in tweets:
        polarity, _ = sentiment(tweet['text'])
        _positive = positive(tweet['text'], threshold=0.1)

        # print sentiment score
        print('\"{}\" has the sentiment score = {}.'.format(tweet['text'], polarity))

        # write text
        file = pos_tweets if _positive is True else neg_tweets
        file.write(tweet['text'] + '\n')

    # clsoe file
    pos_tweets.close()
    neg_tweets.close()