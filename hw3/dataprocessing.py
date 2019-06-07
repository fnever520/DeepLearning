import os
import string
from keras.datasets import imdb
import random
from keras.preprocessing import sequence
from nltk.stem import PorterStemmer
import time
import pandas as pd
import pickle
import matplotlib.pyplot as plt
porter = False

'''
top_words denotes the max_features. the more common the word is, the lower the indices are.

'''

class PrepareData:
    # is_one_file - in case of one text file or text usage
    # top_words - set top words according keras imdb data
    # review_length - set max desirable length of review
    def __init__(self, is_one_file=False, top_words=4500, review_length=600):
        self.wordToIndex = imdb.get_word_index()
        self.top_words = top_words
        self.is_one_file = is_one_file
        self.review_length = review_length

    # if text only - is_text=True, else set path to reviews' dir
    def get_data(self, path='', is_text=False, text='', porter=False):
        data = []
        if self.is_one_file:
            if is_text:
                tokens = self.get_tokens(text)
                data.append(tokens)
            else:
                with open(path, 'r', encoding="utf-8") as file:
                    input_text = file.read()
                    tokens = self.get_tokens(input_text,porter=False)
                    data.append(tokens)
        else:
            files = os.listdir(path)
            for i in range(len(files)):
                with open(path + files[i], 'r', encoding="utf-8") as file:
                    input_text = file.read()
                    tokens = self.get_tokens(input_text,porter=False)
                    data.append(tokens)

        data_norm = []
        # len(data) ==> 12500 for training
        lengths = [len(x) for x in data]
        #with plt.xkcd():
        #    plt.hist(lengths, bins=range(300))
            #plt.show()
        
        for i in range(len(data)):
            one_review = []
            for word in data[i]:
                #reviews_len = [len(x) for x in data[i]]
                #pd.Series(reviews_len).hist()
                #plt.show()
                #pd.Series(reviews_len).describe()
                try:
                    index = self.wordToIndex[word]
                    if index < self.top_words:
                        one_review.append(index)
                except KeyError:
                    pass
            data_norm.append(one_review)
            
        with open('tokenizer.pickle', 'wb') as handle:
            pickle.dump(data_norm, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return data_norm

    # split and process text of review into tokens
    @staticmethod
    def get_tokens(input_text, porter):
        try:
            text = input_text.lower()
            text = text.replace('"', '')
            text = text.replace('<br', '')
            tokens = text.split()
            table = str.maketrans('', '', string.punctuation)
            tokens = [w.translate(table) for w in tokens]
            #print(tokens)
            if (porter):
                ps = PorterStemmer()
                #print("Stemming process....")
                tokens = [ps.stem(w) for w in tokens]
                #print(tokens)
                #time.sleep(15)
            return tokens
        except UnicodeDecodeError:
            print("UnicodeDecodeError in file: ", input_text)
            return ''

    # shuffle pos and neg reviews before feeding to lstm
    @staticmethod
    def binary_shuffle(positive, negative):
        counter_pos = 0
        counter_neg = 0
        x = []
        y = []
        while counter_pos < len(positive) or counter_neg < len(negative):
            rand = random.randrange(0, 2)
            if rand == 0 and counter_neg < len(negative):
                x.append(negative[counter_neg])
                y.append(0)
                counter_neg = counter_neg + 1
            if rand == 1 and counter_pos < len(positive):
                x.append(positive[counter_pos])
                y.append(1)
                counter_pos = counter_pos + 1
        return x, y

    # truncate and pad reviews with zeros
    # use the function - pad_sequence to truncate larger reviewes, and pad smaller reviews with zeros, to a maximum cap of xx
    def review_truncate(self, input_data):
        result = sequence.pad_sequences(input_data, maxlen=self.review_length)
        return result
