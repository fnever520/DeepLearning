import numpy as np
from nltk.stem import PorterStemmer
import re
import os
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

def clean_text(txt):
    # Ensure lowercase text encoding
    txt = str(txt).lower()
    # splits tokens by white space
    tokens = txt.split()
    # remove tokens not encoded in ascii
    isacii = lambda s:len(s) == len(s.encode())
    tokens = [w for w in tokens if isascii(w)]
    # regex for punctuation filtering
    re_punc = re.compile('[%s]' %re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('',w) for w in tokens]
    # remove tokens that are not alphanumeric
    tokens = [w for w in tokens if w.isalnum()]
    # regex for digits filtering
    re_digt = re.compile('[%s]' %re.escape(string.digits))
    # remove digits from each word
    tokens = [re_digt.sub('',w) for w in tokens]
    # filter out stop words
    stop_words = set(stopwords.word('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out long tokens
    tokens = [w for w in tokens if len(w)<30]
    # filter out short tokens
    tokens = [w for w in tokens if len(w)>1]
    # stemming for words
    porter = PorterStemmer()
    tokens = [porter.stem(w) for w in tokens]
    return tokens

def token_to_line(txt, vocab):
    tokens = clean_text(txt)
    # filter by vocabulary
    tokens = [w for w in tokens if w in vocab]
    return ' '.join(tokens)

    
# dataset_path = "aclimdb"
# print("Loading datasets into Memory")





'''
ps = PorterStemmer()
example_words = ["python", "pythoner", "pythoning", "pythoned"]

for x in example_words:
    print(ps.stem(x))

new_test= "it is an important to buy very pythonly while you are pythoning with python"

words = word_tokenize(new_test)
for y in words:
    print(ps.stem(y))

'''