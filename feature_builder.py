import pandas as pd
import time
from multiprocessing import Process
from multiprocessing import Manager
import numpy as np
from pandarallel import pandarallel
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
pandarallel.initialize(nb_workers=16, use_memory_fs=False)

class DiagFeatureBuilder:
    def __init__(self, processed_data):
        self.data = processed_data

    # user-index map: user_name --> i
    def get_user2index(self, data):
        userMap = dict()
        for i, user in enumerate(data.name.unique()):
            userMap[user] = i
        return userMap

    # tweet-index map: tweet_text --> j
    def get_tweet2index(self, data):
        tweetMap = dict()
        for i, tweet in enumerate(data.postTweet.unique()):
            tweetMap[tweet] = i
        return tweetMap

    def build_index_mapping_only(self):
        self.user2index, self.tweet2index = self.get_user2index(self.data), self.get_tweet2index(self.data)


# wget http://nlp.stanford.edu/data/glove.6B.zip
# unzip glove.6B.zip

class TfidfEmbeddingVectorizer(object):
    def __init__(self, X):
        self.word2vec = self.build_w2v(X)
        self.word2weight = None
        if len(self.word2vec) > 0:
            self.dim = len(self.word2vec[next(iter(self.word2vec))])
        else:
            self.dim = 0
        self.fit(X)

    def build_w2v(self, X):
        GLOVE_6B_50D_PATH = "/Users/lijinning/PycharmProjects/Polarization/dataset/w2v/glove/glove.6B.50d.txt"
        glove_small = {}
        all_words = set(w for words in X for w in words)
        with open(GLOVE_6B_50D_PATH, "rb") as infile:
            for line in infile:
                parts = line.split()
                word = parts[0].decode("utf-8")
                if (word in all_words):
                    nums = np.array(parts[1:], dtype=np.float32)
                    glove_small[word] = nums
        return glove_small

    def fit(self, X):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] * self.word2weight[w]
                     for w in words if w in self.word2vec] or
                    [np.zeros(self.dim)], axis=0)
            for words in X
        ])

    def transform_one(self, x):
        return self.transform([x])[0]


if __name__ == "__main__":
    model = TfidfEmbeddingVectorizer([["happy", "deadline", "china"], ["what", "cheap", "chips"]])
    print(model.transform([["happy", "deadline", "china"], ["what", "cheap", "chips"]]))
    print(model.transform_one(["happy", "deadline", "china"]))






