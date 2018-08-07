# packages
import nltk.data
import codecs
import os
import string
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import re
from gensim.models import Word2Vec

# read the data in 
df = pd.read_excel('C:\\Users\\ali\\Downloads\\reddit_posts.xlsx', sheetname='combine_3coders_merge_and_posts')


# remove punctuation
def remove_punctuation(s):
    s = ''.join([i for i in s if i not in frozenset(string.punctuation)])
    return s

df['cleaned'] = df['taskinfo__askhistorians_text'].apply(remove_punctuation)

nltk.download('punkt')
# create the model
model = Word2Vec(df['tokenized_sents'], min_count=1)
print(model)

# import the word2vec file
from gensim.models import KeyedVectors
filename = 'C:\\Users\\ali\\Desktop\\GoogleNews-vectors-negative300.bin'
model1 = KeyedVectors.load_word2vec_format(filename, binary=True)

# find the similar terms to 2 examples
model.similar_by_word('people')
model1.similar_by_word('people')
model.similar_by_word('war')
model1.similar_by_word('war')