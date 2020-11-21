#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pickle

## import data
data_path = './pickle/data_preprocessing.pkl'
fp = open(data_path, 'rb')
x_train_processed, x_test_processed, y_train, y_test, word_index = pickle.load(fp)
fp.close()


embeddings_index = {}
# use Glove word_embeddings to initialise a word dictionary
# with open('./embeddings/glove.6B.50d.txt', 'r', encoding='UTF-8') as embedding_file:
# with open('./embeddings/glove.6B.100d.txt', 'r', encoding='UTF-8') as embedding_file:
# with open('./embeddings/glove.6B.200d.txt', 'r', encoding='UTF-8') as embedding_file:
with open('./embeddings/glove.6B.300d.txt', 'r', encoding='UTF-8') as embedding_file:
    for line in embedding_file:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
embedding_file.close()
print('Total %s word vectors.' % len(embeddings_index))


# EMBEDDING_DIM = 51 # Glove 50-dimension
# EMBEDDING_DIM = 101 # Glove 100-dimension
# EMBEDDING_DIM = 201 # Glove 200-dimension
EMBEDDING_DIM = 301 # Glove 300-dimension
# embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))  # random between (number of words, embedd dimensions)
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)

    if word == 'P':
        pronoun_vector = 1
    elif word == 'A':
        pronoun_vector = 2
    elif word == 'B':
        pronoun_vector = 3
    else:
        pronoun_vector = 0

    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = np.append(embedding_vector, pronoun_vector)
    else:
        # embedding_matrix[i] = np.append(np.zeros(50, dtype='float32'), pronoun_vector)
        # embedding_matrix[i] = np.append(np.zeros(100, dtype='float32'), pronoun_vector)
        # embedding_matrix[i] = np.append(np.zeros(200, dtype='float32'), pronoun_vector)
        embedding_matrix[i] = np.append(np.zeros(300, dtype='float32'), pronoun_vector)


import pickle
data = (x_train_processed, x_test_processed, y_train, y_test, word_index, embedding_matrix)
# fp = open('./pickle/data_embeddings.pkl', 'wb')
# fp = open('./pickle/data_embeddings_100d.pkl', 'wb')
# fp = open('./pickle/data_embeddings_200d.pkl', 'wb')
fp = open('./pickle/data_embeddings_300d.pkl', 'wb')

pickle.dump(data, fp)
fp.close()

