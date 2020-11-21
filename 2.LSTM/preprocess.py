import pandas as pd
import numpy as np

# test = pd.read_csv("../data/test_stage_1.tsv",sep = '\t')
# print(test.head())
# sub = pd.read_csv("../data/sample_submission_stage_1.csv",sep = '\t')
# print(sub.head())

#load data
gap_test = pd.read_csv("../data/gap-test.tsv",sep = '\t')
gap_valid = pd.read_csv('../data/gap-validation.tsv', sep = '\t')
train = pd.concat([gap_test, gap_valid], ignore_index=True, sort=False)
print(gap_test.shape, gap_valid.shape, train.shape)
test = pd.read_csv('../data/gap-development.tsv', sep = '\t')
data_all = pd.concat([train, test], ignore_index=True, sort=False)
print(test.shape, data_all.shape)



def insert_tag(row):
    to_be_inserted = sorted([  # Simple sort, using the offset name in descending order (P-B-A)
                            (row['A-offset'], ' [A] '),
                            (row['B-offset'], ' [B] '),
                            (row['Pronoun-offset'], ' [P] ')
                             ], key=lambda x: x[0], reverse=True)
    text = row['Text']
    for offset, tag in to_be_inserted:
        text = text[:offset] + tag + text[offset:]  # insert the TAG (A/B/P) in front of the actual word

    return text

data_all['Text'] = data_all.apply(insert_tag, axis = 1)
# print(data_all.shape)
# print(data_all['Text'][0])


# word segmentation
def doc2word(data):
    doc2words = []
    for line in data:
        list = line.split(' ')
        doc2words.append(list)
    return doc2words
doc2words = doc2word(data_all['Text'])

# delete all the symbols surrounding tokens
all_flags=[',', '.', '!', '?', ';', "''", "`",'(',')',':']
def text_clean(doc2words):
    text = []
    for line in doc2words:
        temp_line = [] # save all the words in one row into a temp list
        for word in line:
            for flag in all_flags:
                word = word.strip(flag)
            temp_line.append(word)
        text.append(temp_line)
    return text

text_preprocessed = text_clean(doc2words)

x_train = text_preprocessed[:train.shape[0]]
x_test = text_preprocessed[train.shape[0]:]

# Tokenize the text
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
def build_word_tokenizer(text):
    vocabulary = []
    for document in text:
        for word in document:
            vocabulary.append(word)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([vocabulary])  # generate a token dictionary
    return tokenizer
tokenizer = build_word_tokenizer(x_train)
# texts_to_sequences(texts) converts words to a vector form with their index in dictionary
# shape: [len(texts),len(text)] -- (number of document, length of each document)
x_train_token = tokenizer.texts_to_sequences(x_train)
x_test_token = tokenizer.texts_to_sequences(x_test)

# tokenizer = build_word_tokenizer(x_train)
# x_train_token = tokenizer.texts_to_sequences(x_train)
# x_test_token = tokenizer.texts_to_sequences(x_test)


# pad the sentence
maxLen = len(max(x_train, key=len)) # Get the sample according to its max length, and then take the length
# padding the sequence to the maxlen length, the padding value has pre|post, and value specifies what value to fill in
x_train_processed = pad_sequences(x_train_token, maxlen=maxLen)
x_test_processed = pad_sequences(x_test_token, maxlen=maxLen)

word_index = tokenizer.word_index  # output each word in the Tokenizer dictionary and its corresponding index

# process labels
def row_to_y(row):
    if row.loc['A-coref']:
        return 0
    if row.loc['B-coref']:
        return 1
    return 2

# execute 'row to y' function by axis 1, get label list
y_train = train.apply(row_to_y, axis = 1)
y_test = test.apply(row_to_y, axis = 1)

# converts the category vector to a binary (only 0/1) 'singly hot' matrix type representation.
# N classes are n by n matrices
from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# use the pickle package to do serialization operations, and save the data/model infomation of the program to a file for permanent storage
import pickle
data = (x_train_processed, x_test_processed, y_train, y_test, word_index)
fp = open('./pickle/data_preprocessing.pkl', 'wb')
pickle.dump(data, fp)
fp.close()

text = (x_train, x_test)
fp = open('./pickle/cleaned_text.pkl', 'wb')
pickle.dump(text, fp)
fp.close()

