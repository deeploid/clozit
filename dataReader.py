#!/usr/bin/python
import os, re, cPickle
import numpy as np
from collections import defaultdict
import gensim

def load_word2vec(name='GoogleNews-vectors-negative300.bin.gz'):
    return gensim.models.Word2Vec.load_word2vec_format(name,binary=True)

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip().lower()

def embedder(l):
    global model
    return np.asarray([ model[str(word)] for word in l.split() if str(word) in model])

def load_data():
    ###############
    ## Load Data ##
    ###############
    data_folder = "resource/"
    w2v_file = data_folder+"GoogleNews-vectors-negative300.bin"
    print('loading word2vec vectors...')
    w2v = load_word2vec(name=w2v_file)
    print("word2vec loaded!")
    return w2v

model = load_data()
def next_batch(filename):
    global model
    with open(filename) as f:
        for line in f:
            if line in '': continue
            # Get doc
            doc = line.split('///')[0]
            doc = embedder( clean_str(doc,False) )
            # Get question
            question = line.split('///')[1]
            question = embedder( clean_str(question,False) )
            # Get options
            options = line.split('///')[2].split(',')
            yield (doc,question,options)

if __name__=="__main__":
    # Sample
    sent = 'LeBron had spent the weekend watching old Muhammad Ali fights, in awe at the champ\'s perseverance.'
    
    ###############
    ## Load Data ##
    ###############
    data_folder = "resource/"
    w2v_file = data_folder+"GoogleNews-vectors-negative300.bin"
    print('loading word2vec vectors...')
    w2v = load_word2vec(name=w2v_file)
    print("word2vec loaded!")

    ##################
    ## Test modules ##
    ##################
    # getting embedding word list
    e_list = embedder( clean_str(sent,True), w2v )
    print(e_list)

    #print("num words already in word2vec: " + str(len(w2v)))
