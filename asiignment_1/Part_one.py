import sys
assert sys.version_info[0] == 3
assert sys.version_info[1] >=5

from gensim.models import KeyedVectors
from gensim.test.utils import datapath

import pprint
import matplotlib.pyplot as plt 
plt.rcParams['figure.figsize'] = [10,5]

from nltk.corpus import reuters
import numpy as np 
import scipy as sp 
import random

from sklearn.decomposition import TruncatedSVD, PCA

START_TOKEN = '<STARK>'
END_TOKEN = '<END>'

np.random.seed(0)
random.seed(0)


def read_corpus(category = 'crude'):
    files = reuters.fileids(category)
    return [[START_TOKEN] + [w.lower() for w in list(reuters.words(f))] + [END_TOKEN] for f in files]

def distinct_words(corpus):
    corpus_words = []
    num_corpus_words = -1
    reposity_corpus = set()

    for sentence in corpus:
        reposity_corpus  = reposity_corpus.union(set(sentence))

    corpus_words = list(reposity_corpus)
    num_corpus_words = len(corpus_words)
    corpus_words = sorted(corpus_words)

    return corpus_words, num_corpus_words

def compute_co_occurrence_matrix(corpus,windows_size = 4):
    words, num_words = distinct_words(corpus)
    M = np.zeros([num_words,num_words])

    index_list = [i for i in range(-windows_size//2, windows_size//2)]

    ids = [i for i in range(num_words)]
    temp = list(zip(words,ids))
    word2ids = dict(temp)
    for words in corpus:
        length = len(words)
        for i in range(length):
            for j in index_list:
                z = i + j
                if z < 0 or z > (length-1):
                    continue
                else:
                    w_i = word2ids[words[i]]
                    w_j = word2ids[words[z]]
                    M[w_i,w_j] += 1
                    if w_i != w_j:
                        M[w_j,w_i] += 1

    return M, word2ids,words

def reduce_to_k_dim(M,k=2):
    n_iters = 10
    SVD_model = TruncatedSVD(k,algorithm = "randomized",n_iter = n_iters)
    reduce_M = SVD_model.fit_transform(M)
    print(reduce_M)
    return reduce_M

def plot_vec(reduce_M,word2ids, words):
    words = list(set(words))
    for word in words:
        pos = reduce_M[word2ids[word],:]

        plt.scatter(pos[0],pos[1],s = 5,alpha=0.4)
        plt.annotate(word, xy=pos,xytext = [pos[0]+0.1, pos[1]+0.1])
    plt.show()
    plt.savefig('./fig.jpg',dpi=300)
    

if __name__ == "__main__":
    sentence = read_corpus()
    test_s = sentence[0]

    # ---------------------
    # Run this sanity check
    # Note that this not an exhaustive check for correctness.
    # ---------------------

    # Define toy corpus
    test_corpus = ["{} All that glitters isn't gold {}".format(START_TOKEN, END_TOKEN).split(" "), "{} All's well that ends well {}".format(START_TOKEN, END_TOKEN).split(" ")]
    #test_corpus_words, num_corpus_words = distinct_words(test_corpus)
    M,word2ids,words=compute_co_occurrence_matrix(test_corpus,1)
    reduce_M = reduce_to_k_dim(M)
    plot_vec(reduce_M,word2ids,words)
    print(words)