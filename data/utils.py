from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import math
import os
import random
import sys
import time
import tensorflow.python.platform


def gunzip_file(gz_path, new_path):
    if not (gfile.Exists(new_path)):
        print("Unpacking %s to %s" % (gz_path, new_path))
        with gzip.open(gz_path, "rb") as gz_file:
            with open(new_path, "wb") as new_file:
                for line in gz_file:
                    new_file.write(line)
    else:
        print("It is already there")
        
        
def get_data(directory):
    train_path = os.path.join(directory,"train")
    #os.mkdir(train_path)
    corpus_file=os.path.join(directory,"train.tar")
    print("Extracting training  files " )
    with tarfile.open(corpus_file, "r") as corpus_tar:
        corpus_tar.extractall(directory)
        gunzip_file(directory + "/q_train.gz", train_path + "_q")
        gunzip_file(directory + "/a_train.gz", train_path + "_a")
        #creating vocabulary
        print(" checking for  vocabulries  ")
        if not ( gfile.Exists(os.path.join(directory, "vocab%d_q" % q_vocabulary_size)) and
                 gfile.Exists(os.path.join(directory, "vocab%d_a" % a_vocabulary_size))):
            print("creating vocabulries")
            q_vocab_path, a_vocab_path, _ = create_voca(directory)
        else:
            q_vocab_path=os.path.join(directory, "vocab%d_q" % q_vocabulary_size)
            a_vocab_path=os.path.join(directory, "vocab%d_a" % a_vocabulary_size)
            print("Vocabulries are already there.")        
        #creating token ids
        print("Tokenizing data")
        q_train_ids_path = train_path + (".qids%d" % q_vocabulary_size)
        a_train_ids_path = train_path + (".aids%d" % a_vocabulary_size)        
        data_to_token_ids(train_path + "_q", q_train_ids_path, q_vocab_path)
        data_to_token_ids(train_path + "_a", a_train_ids_path, a_vocab_path)
    return(q_train_ids_path,a_train_ids_path,q_vocab_path,a_vocab_path)
    
    
def data_to_token_ids(data_path, target_path, vocabulary_path, normalize_digits=False):
    #print(vocabulary_path)
    if not gfile.Exists(target_path):
        print("Tokenizing data in %s" % data_path)
        vocab, _ = initialize_vocabulary(vocabulary_path)
        with gfile.GFile(data_path, mode="r") as data_file:
            with gfile.GFile(target_path, mode="w") as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 100000 == 0:
                        print("  tokenizing line %d" % counter)
                    token_ids = sentence_to_token_ids(line, vocab, normalize_digits)
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")
                    
def sentence_to_token_ids(sentence, vocabulary,tokenizer=None, normalize_digits=True):
    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]
    return [vocabulary.get(re.sub(_DIGIT_RE, b"0", w), UNK_ID) for w in words]
    
def initialize_vocabulary(vocabulary_path):
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="rb") as f:
            f.readlines()
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)
        
def create_voca(data_dir):
    q_vocab_path = os.path.join(data_dir, "vocab%d_q" % q_vocabulary_size)
    a_vocab_path = os.path.join(data_dir, "vocab%d_a" % a_vocabulary_size)
    create_vocabulary(q_vocab_path, data_dir + "/train_q", q_vocabulary_size)
    create_vocabulary(a_vocab_path, data_dir + "/train_a", a_vocabulary_size)
    print("vocabularies are created in %s " % data_dir + "file  is %s " % q_vocab_path)
    return (q_vocab_path,a_vocab_path)
    
    
def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size, normalize_digits=False):
    tokenizer=None
    vocab = {}
    with gfile.GFile(data_path, mode="rb") as f:
        counter = 0
        for line in f:
            counter += 1
            if counter % 100000 == 0:
                print("  processing line %d" % counter)
            tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
            for w in tokens:
                word = re.sub(_DIGIT_RE, b"0", w) if normalize_digits else w
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1
        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]
        with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
            for w in vocab_list:
                vocab_file.write(w + b"\n")
                
def basic_tokenizer(sentence):
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
    return [w for w in words if w]
