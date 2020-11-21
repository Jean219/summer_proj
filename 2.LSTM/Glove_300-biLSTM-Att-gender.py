#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.models import Model, Sequential
from keras.layers import Dense, Input, Dropout, LSTM, Bidirectional, Activation, GRU, SpatialDropout1D, GlobalMaxPool1D, \
    Flatten, merge, RepeatVector, Permute, Lambda, Multiply, multiply
from keras.layers.embeddings import Embedding

import keras.backend as K
import pickle


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
    
## import data
# data_path = './pickle/data_embeddings_gender.pkl'
data_path = './pickle/data_embeddings_gender_300d.pkl'
fp = open(data_path, 'rb')
x_train_processed, x_test_processed, y_train, y_test, word_index, embedding_matrix = pickle.load(fp)
fp.close()

data_path = './pickle/cleaned_text.pkl'
fp = open(data_path, 'rb')
x_train, x_test = pickle.load(fp)
fp.close()

# EMBEDDING_DIM = 54
EMBEDDING_DIM = 304
maxLen = len(max(x_train, key=len))
units = 128

_input = Input(shape=(maxLen,), dtype='float32')
# get the embedding layer
embedding_layer = Embedding(input_dim=len(word_index) + 1,
                            output_dim=EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=maxLen,
                            trainable=False
                            )
                                                    
embedded = embedding_layer(_input)

                                                  
activations = Bidirectional(LSTM(int(units/2), return_sequences = True), name='bidirectional_lstm')(embedded)
print(activations.get_shape())  # (None, 229, 256)
# activations = LSTM(units, return_sequences = True)(embedded)

# compute importance for each step
attention = Dense(1, activation='tanh')(activations)
# print(attention.get_shape())  # (None, 229, 1)
attention = Flatten()(attention)
# print(attention.get_shape())  # (None, 229)
attention = Activation('softmax')(attention)
# print(attention.get_shape())  # (None, 229)
attention = RepeatVector(units)(attention)
# print(attention.get_shape())  # (None, 128, 229)
attention = Permute([2, 1])(attention)
# print(attention.get_shape())  # (None, 229, 128)


# apply the attention
# sent_representation = merge([activations, attention], mode='mul')
# sent_representation = Multiply()([activations, attention])
sent_representation = multiply([activations, attention])
sent_representation = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(units,))(sent_representation)
probabilities = Dense(units=3, activation='softmax', name='output')(sent_representation)



model = Model(inputs=_input, outputs=probabilities)
model.summary()
# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy",f1_m,precision_m, recall_m])



# fit the model
history = model.fit(x_train_processed, y_train, epochs = 5, batch_size = 256, shuffle = True, validation_data=(x_test_processed, y_test))

# evaluate the model
loss, accuracy, f1_score, precision, recall = model.evaluate(x_test_processed, y_test, verbose=0)

print('loss:',loss)
print('accuracy:',accuracy)
print('f1_score:',f1_score)
print('precision:',precision)
print('recall:',recall)

import numpy as np
from numpy import *
import pandas as pd

test = pd.read_csv('../data/gap-development.tsv', sep = '\t')
y_true = np.argmax(y_test, axis=1)  # find the index of the maximum value in each line
y_test_pred = model.predict(x_test_processed, verbose=0)

y_true_df = pd.DataFrame(y_true)
y_test_pred_df = pd.DataFrame(y_test_pred)
df = pd.concat([test['ID'],y_true_df,y_test_pred_df],axis=1)
df.columns = 'ID y_true p_a p_b p_neither '.split()
df.to_excel('../predicted_data_Glove 300 + biLSTM + attention + gender features.xlsx',sheet_name='lstm5')