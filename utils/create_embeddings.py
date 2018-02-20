import numpy as np
import pandas as pd
import _pickle as cPickle
from collections import defaultdict
import re

from bs4 import BeautifulSoup

import sys
import os

from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot

os.environ['KERAS_BACKEND']='theano'

from config import *


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string.decode("utf-8"))
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

data_train = pd.read_csv(DATA_PATH, sep='\t')
print(data_train.shape)

texts = []
for idx in range(data_train.review.shape[0]):
    text = BeautifulSoup(data_train.review[idx], "lxml")
    text = clean_str(text.get_text().encode('ascii','ignore'))
    texts.append(text)

sentences = []
for text in texts:
    sentences.append(text.split(' '))


model = Word2Vec(sentences, min_count=1)
# summarize the loaded model
print(model)
# access vector for one word
print(model['sentence'])
# save model
model.wv.save_word2vec_format('tweet_embeddings.100d.txt', binary=False)
