#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pickle
import csv
from nltk.corpus import wordnet

## import data
data_path = './pickle/data_preprocessing.pkl'
fp = open(data_path, 'rb')
x_train_processed, x_test_processed, y_train, y_test, word_index = pickle.load(fp)
fp.close()


# load the word_embedding dictionary
embeddings_index = {}
# use Glove word_embeddings to initialise a word dictionary
with open('./embeddings/glove.6B.300d.txt', 'r', encoding='UTF-8') as embedding_file:
# with open('./embeddings/glove.6B.50d.txt', 'r', encoding='UTF-8') as embedding_file:
    for line in embedding_file:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
embedding_file.close()
print('Total %s word vectors.' % len(embeddings_index))


# load the name gender frequency
gender_frequency = {}
# Gender by Name, DATASET BY DEREK HOWARD
with open('../1.traditional_classifier/resources/name_gender.csv', 'r', encoding='UTF-8') as gender_file:
    gender_data = csv.reader(gender_file)
    # gender_data = csv.DictReader(gender_file)
    for row in gender_data:
        gender_frequency[row[0].lower()] = row[1]
gender_file.close()
for male in ['he','him','his']:
    gender_frequency[male] = 'M'
for female in ['she','her','hers']:
    gender_frequency[female] = 'F'


# extract semantic feature
def _extract_sem_class(word):
    synset = None
    word = word.lower()

    # if the ne is a pronoun, replace it by a specific name
    if word in ['his', 'he', 'him']:
        word = 'matthew'
    elif word in ['she', 'her', 'hers']:
        word = 'maria'

    for s in wordnet.synsets(word):
        if s.name().split('.')[1] == 'n':
            synset = s

    # check if set is found
    if synset is None:
        return 0,0

    # # extract all hypernyms
    hypers = list(synset.closure(lambda s: s.hypernyms()))
    all_hypernyms = [hyper.name() for hyper in hypers]

    semantic_classes = {'person', 'male', 'female'}
    # return first class found (= most specific class)
    for hypernym in all_hypernyms:
        if hypernym.split('.')[0] not in semantic_classes:
            return 0,0
        elif hypernym.split('.')[0] == 'person':
            return 0,1
        elif hypernym.split('.')[0] == 'male':
            return 0,2
        elif hypernym.split('.')[0] == 'female':
            return 0,3
    # class not found
    return 0,0



# extract gender feature
def _extract_gender(ne_string):
    try:
        gender = gender_frequency[ne_string.lower()]
    except KeyError:
        return 0

    # index 0 is masculine
    if gender == 'M':
        return 1
    # index 1 is feminine
    elif gender == 'F':
        return 2
    # index 2 is neuter
    else:
        return 0



# EMBEDDING_DIM = 54 # Glove 50-dimension
EMBEDDING_DIM = 304 # Glove 300-dimension
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

    sem_class_vector,sem_person_vector = _extract_sem_class(word)
    gender_vector = _extract_gender(word)

    vector_list = [pronoun_vector, sem_class_vector, sem_person_vector, gender_vector]
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = np.append(embedding_vector, vector_list)
    else:
        embedding_matrix[i] = np.append(np.zeros(300, dtype='float32'), vector_list)

# print(embedding_matrix[100])
import pickle
data = (x_train_processed, x_test_processed, y_train, y_test, word_index, embedding_matrix)
# fp = open('./pickle/data_embeddings_gender.pkl', 'wb')
fp = open('./pickle/data_embeddings_gender_300d.pkl', 'wb')
pickle.dump(data, fp)
fp.close()

