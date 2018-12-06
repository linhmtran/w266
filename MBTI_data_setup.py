from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time

import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from w266_common import utils, vocabulary

# function to clean and tokenize sentence ["Hello world."] into list of words ["hello world"]
def clean(sentence):
    ignore_words = []
    words = re.sub("[^\w]", " ",  sentence).split() #nltk.word_tokenize(sentence)
#     words = sentence.split() #nltk.word_tokenize(sentence)
    words_cleaned = sentence.lower()
    return words_cleaned

def splitPosts(df):
    # split posts per users into separate sentences
    post = []
    utype = []
    user = []

    for index, row in df.iterrows():
        posts = row['posts'].split('|||')
        posts_clean = []
        for sentence in posts:
            posts_clean.append(clean(sentence))
        post.extend(posts_clean)
    #     post.extend(posts)
        utype.extend([row['type'] for i in range(len(posts))])
        user.extend([index for i in range(len(posts))])

    short_posts = pd.DataFrame({"user": user,"type": utype,"post": post})
    print(short_posts.shape)
    return np.array(short_posts['post']), np.array(short_posts['type']), np.array(short_posts['user'])

def train_test_split(x,y,test_pct,seed=88):
    return train_test_split(x, y, test_size=test_pct, random_state=seed)
    
def full_vocab_canon(x):
    # Build a vocabulary (V size is defaulted to full text) for train corpus
    vocab_mbti = vocabulary.Vocabulary((utils.canonicalize_word(w) for w in x))
    print("Full Vocab Built, size: ", vocab_mbti.size)
    return vocab_mbti.size, vocab_mbti

def canonicalize(x):
    # tokenize and canonicalize train and test sets
    x_c = []
    for post in post_train:
        x_c.append(vocab_mbti.words_to_ids(x.split()))
    return x_c


def one_hot_label(mbti_type, label_train, label_test):
    #create integer classifiers as 1 hot
    keys = list(set(mbti_type))
    values = list(range(len(keys)))
    label_map = dict(zip(keys, values))

    y_train_id = np.array([label_map[label] for label in label_train])
    one_hot_train = np.zeros((len(label_train),16),dtype=int)
    one_hot_train[np.arange(len(label_train)), y_train_id] = 1
    y_train = one_hot_train

    y_test_id = np.array([label_map[label] for label in label_test])
    one_hot_test = np.zeros((len(label_test),16),dtype=int)
    one_hot_test[np.arange(len(label_test)), y_test_id] = 1
    y_test = one_hot_test
    
    return y_train, y_test

def label_to_id(mbti_type, label_train, label_test):
        #create integer classifiers as 1 hot
    keys = list(set(mbti_type))
    values = list(range(len(keys)))
    label_map = dict(zip(keys, values))

    y_train_id = np.array([label_map[label] for label in label_train])

    y_test_id = np.array([label_map[label] for label in label_test])
    
    return y_train_id, y_test_id