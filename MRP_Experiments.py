import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import nltk
import string
import numpy as np
import re
import snowballstemmer
from nltk.corpus import stopwords
from gensim.models import Word2Vec
 
# read the data in 
 df = pd.read_excel('C:\\Users\\ali\\Downloads\\reddit_posts.xlsx', sheetname='combine_3coders_merge_and_posts')

# data preprocessing
def clean_text(text):
    
    ## Remove puncuation
    text = text.translate(string.punctuation)
    
    ## Convert words to lower case and split them
    text = text.lower().split()
    
    ## Remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]
    
    text = " ".join(text)

    ## Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
        ## Stemming
    #text = text.split()
    #stemmer = snowballstemmer.stemmer('english')
    #stemmed_words = [stemmer.stemWord(word) for word in text]
    #text = " ".join(stemmed_words)
    
    return text
	
# apply the above function to df['text']

df['cleaned'] = df['taskinfo__askhistorians_text'].map(lambda x: clean_text(x))

# also removing punctuation
def remove_punctuation(s):
    s = ''.join([i for i in s if i not in frozenset(string.punctuation)])
    return s

df['cleaned'] = df['cleaned'].apply(remove_punctuation)


# create models to compare word similarity create by training data and pre-computed word embedding
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

#create a list of posts
texts = []
for i in range(0, 1208): #start from 1, to leave out row 0
    texts.append(df['cleaned'][i]) #extract from first col
    

# lets take 80% data as training and remaining 20% for test.
train_size = int(len(texts) * .80)
 
train_posts = texts[:train_size]
train_labels = labels[:train_size]

test_posts = texts[train_size:]
test_labels = labels[train_size:]

# training the models separately for each label
label_c3 = np.array(train_labels['c3'])
label_c5 = np.array(train_labels['c5'])
label_c6 = np.array(train_labels['c6'])
label_c7 = np.array(train_labels['c7'])

# tokenize the texts and pad with zero if the length is shorter than 200
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
max_words = 10000
maxlen=200
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_posts)
sequences = tokenizer.texts_to_sequences(train_posts)
word_index = tokenizer.word_index
data = pad_sequences(sequences, maxlen=maxlen)

# reading Glove word embedding + creating word index and word vectors
glove_dir = 'C:/Users/ali/Downloads/glove.6B'
embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt') , encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

embedding_dim = 100
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
			
# CNN models
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate

import warnings
from scipy import mean
warnings.filterwarnings(action = 'once')

# set parameters:
filters = 250
kernel_size = 3

#set seed for reproducibility
seed = 7
def conv1d_model():
    model = Sequential()
    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(max_words, embedding_dim,input_length=maxlen))
    model.add(Dropout(0.2))
    # we add a Convolution1D, which will learn filters
    # word group filters of size filter_length:
    model.add(Conv1D(filters,kernel_size,padding='valid',activation='relu',strides=1))
    # we use max pooling:
    model.add(GlobalMaxPooling1D())
    # We add a vanilla hidden layer:
    model.add(Dense(32))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(1, activation='sigmoid'))
    model.layers[0].set_weights([embedding_matrix])
    #freezing the embedding layer form updating
    model.layers[0].trainable = False
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
    return model

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_micro': 'recall_macro'}

#c3
class_weight = {0: 1.0/616.0,
                1: 1.0/592.0}
conv1d_estimator_c3 = KerasClassifier(build_fn=conv1d_model, epochs=10, batch_size=32, verbose=0, class_weight=class_weight)
results_conv1d_c3 = cross_validate(conv1d_estimator_c3, data, label_c3, cv=kfold, scoring= scoring)
print(results_conv1d_c3)
print('')

for st,vals in results_conv1d_c3.items():
    print("Average for {} is {}".format(st,mean(vals)))
print('')    

#c5
class_weight = {0: 1.0/1004.0,
                1: 1.0/204.0}
conv1d_estimator_c5 = KerasClassifier(build_fn=conv1d_model, epochs=10, batch_size=32, verbose=0, class_weight=class_weight)
results_conv1d_c5 = cross_validate(conv1d_estimator_c5, data, label_c5, cv=kfold, scoring= scoring)
print(results_conv1d_c5)
print('')

for st,vals in results_conv1d_c5.items():
    print("Average for {} is {}".format(st,mean(vals)))
print('') 

#c6
class_weight = {0: 1.0/934.0,
                1: 1.0/274.0}
conv1d_estimator_c6 = KerasClassifier(build_fn=conv1d_model, epochs=10, batch_size=32, verbose=0, class_weight=class_weight)
results_conv1d_c6 = cross_validate(conv1d_estimator_c6, data, label_c6, cv=kfold, scoring= scoring)
print(results_conv1d_c6)
print('')

for st,vals in results_conv1d_c6.items():
    print("Average for {} is {}".format(st,mean(vals)))
print('') 

#c7
class_weight = {0: 1.0/948.0,
                1: 1.0/260.0}

conv1d_estimator_c7 = KerasClassifier(build_fn=conv1d_model, epochs=10, batch_size=32, verbose=0, class_weight=class_weight)
results_conv1d_c7 = cross_validate(conv1d_estimator_c7, data, label_c7, cv=kfold, scoring= scoring)
print(results_conv1d_c7)
print('')

for st,vals in results_conv1d_c7.items():
    print("Average for {} is {}".format(st,mean(vals)))
print('') 

#visualizing cnn model
model = Sequential()
# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_words, embedding_dim,input_length=maxlen))
model.add(Dropout(0.2))
# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
model.add(Conv1D(filters,kernel_size,padding='valid',activation='relu',strides=1))
# we use max pooling:
model.add(GlobalMaxPooling1D())
# We add a vanilla hidden layer:
model.add(Dense(32))
model.add(Dropout(0.2))
model.add(Activation('relu'))
# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1, activation='sigmoid'))
model.summary()

#lstm
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

import warnings
from scipy import mean
warnings.filterwarnings(action = 'once')

seed = 7
def lstm_model():
    model = Sequential()
    model.add(Embedding(max_words, embedding_dim,input_length=maxlen))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.layers[0].set_weights([embedding_matrix])
    #freezing the embedding layer form updating
    model.layers[0].trainable = False
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
    return model

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_micro': 'recall_macro'}

#c3
class_weight = {0: 1.0/616.0,
                1: 1.0/592.0}

lstm_estimator_c3 = KerasClassifier(build_fn=lstm_model, epochs=10, batch_size=32, verbose=0, class_weight=class_weight)
results_lstm_c3 = cross_validate(lstm_estimator_c3, data, label_c3, cv=kfold, scoring= scoring)
print(results_lstm_c3)
print('')

for st,vals in results_lstm_c3.items():
    print("Average_c3 for {} is {}".format(st,mean(vals)))
print('') 


#c5
class_weight = {0: 1.0/1004.0,
                1: 1.0/204.0}

lstm_estimator_c5 = KerasClassifier(build_fn=lstm_model, epochs=10, batch_size=32, verbose=0, class_weight=class_weight)
results_lstm_c5 = cross_validate(lstm_estimator_c5, data, label_c5, cv=kfold, scoring= scoring)
print(results_lstm_c5)
print('')

for st,vals in results_lstm_c5.items():
    print("Average_c5 for {} is {}".format(st,mean(vals)))
print('') 

#c6
class_weight = {0: 1.0/934.0,
                1: 1.0/274.0}

lstm_estimator_c6 = KerasClassifier(build_fn=lstm_model, epochs=10, batch_size=32, verbose=0, class_weight=class_weight)
results_lstm_c6 = cross_validate(lstm_estimator_c6, data, label_c6, cv=kfold, scoring= scoring)
print(results_lstm_c6)
print('')

for st,vals in results_lstm_c6.items():
    print("Average_c6 for {} is {}".format(st,mean(vals)))
print('') 

#c7
class_weight = {0: 1.0/948.0,
                1: 1.0/260.0}

lstm_estimator_c7 = KerasClassifier(build_fn=lstm_model, epochs=10, batch_size=32, verbose=0, class_weight=class_weight)
results_lstm_c7 = cross_validate(lstm_estimator_c7, data, label_c7, cv=kfold, scoring= scoring)
print(results_lstm_c7)
print('')

for st,vals in results_lstm_c7.items():
    print("Average_c7 for {} is {}".format(st,mean(vals)))
print('') 

#visualizing lstm model
model = Sequential()
model.add(Embedding(max_words, embedding_dim,input_length=maxlen))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.summary()

#lstm + mean pooling
from keras.layers import AveragePooling1D, TimeDistributed, Flatten
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

import warnings
from scipy import mean
warnings.filterwarnings(action = 'once')

seed = 7
def lstm_mp_model():
    model = Sequential()
    model.add(Embedding(max_words, embedding_dim,input_length=maxlen))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.add(AveragePooling1D())
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.layers[0].set_weights([embedding_matrix])
    #freezing the embedding layer form updating
    model.layers[0].trainable = False
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
    return model


kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_micro': 'recall_macro'}

#c3
class_weight = {0: 1.0/616.0,
                1: 1.0/592.0}
lstm_mp_estimator_c3 = KerasClassifier(build_fn=lstm_mp_model, epochs=10, batch_size=32, verbose=0, class_weight=class_weight)
results_lstm_mp_c3 = cross_validate(lstm_mp_estimator_c3, data, label_c3, cv=kfold, scoring= scoring)
print(results_lstm_mp_c3)
print('')

for st,vals in results_lstm_mp_c3.items():
    print("Average_c3 for {} is {}".format(st,mean(vals)))
print('') 

#c5
class_weight = {0: 1.0/1004.0,
                1: 1.0/204.0}

lstm_mp_estimator_c5 = KerasClassifier(build_fn=lstm_mp_model, epochs=10, batch_size=32, verbose=0, class_weight=class_weight)
results_lstm_mp_c5 = cross_validate(lstm_mp_estimator_c5, data, label_c5, cv=kfold, scoring= scoring)
print(results_lstm_mp_c5)
print('')

for st,vals in results_lstm_mp_c5.items():
    print("Average_c5 for {} is {}".format(st,mean(vals)))
print('')

#c6
class_weight = {0: 1.0/934.0,
                1: 1.0/274.0}
lstm_mp_estimator_c6 = KerasClassifier(build_fn=lstm_mp_model, epochs=10, batch_size=32, verbose=0, class_weight=class_weight)
results_lstm_mp_c6 = cross_validate(lstm_mp_estimator_c6, data, label_c6, cv=kfold, scoring= scoring)
print(results_lstm_mp_c6)
print('')

for st,vals in results_lstm_mp_c6.items():
    print("Average_c6 for {} is {}".format(st,mean(vals)))
print('')

#c7
class_weight = {0: 1.0/948.0,
                1: 1.0/260.0}

lstm_mp_estimator_c7 = KerasClassifier(build_fn=lstm_mp_model, epochs=10, batch_size=32, verbose=0, class_weight=class_weight)
results_lstm_mp_c7 = cross_validate(lstm_mp_estimator_c7, data, label_c7, cv=kfold, scoring= scoring)
print(results_lstm_mp_c7)
print('')

for st,vals in results_lstm_mp_c7.items():
    print("Average_c7 for {} is {}".format(st,mean(vals)))
print('')

#visualizing lstm_mp model
model = Sequential()
model.add(Embedding(max_words, embedding_dim,input_length=maxlen))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.add(AveragePooling1D())
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.summary()

#cnn_lstm
from keras.layers import Dropout, Conv1D, MaxPooling1D
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

import warnings
from scipy import mean
warnings.filterwarnings(action = 'once')

# set parameters:
filters = 250
kernel_size = 3

seed = 7

def cnn_lstm_model():
    model = Sequential()
    model.add(Embedding(max_words, embedding_dim,input_length=maxlen))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters,kernel_size,padding='valid',activation='relu',strides=1))
    model.add(MaxPooling1D())
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.layers[0].set_weights([embedding_matrix])
    #freezing the embedding layer form updating
    model.layers[0].trainable = False
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
    return model

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_micro': 'recall_macro'}

#c3
class_weight = {0: 1.0/616.0,
                1: 1.0/592.0}
cnn_lstm_estimator_c3 = KerasClassifier(build_fn=cnn_lstm_model, epochs=10, batch_size=32, verbose=0, class_weight=class_weight)
results_cnn_lstm_c3 = cross_validate(cnn_lstm_estimator_c3, data, label_c3, cv=kfold, scoring= scoring)
print(results_cnn_lstm_c3)
print('')

for st,vals in results_cnn_lstm_c3.items():
    print("Average_c3 for {} is {}".format(st,mean(vals)))
print('') 

#c5
class_weight = {0: 1.0/1004.0,
                1: 1.0/204.0}
cnn_lstm_estimator_c5 = KerasClassifier(build_fn=cnn_lstm_model, epochs=10, batch_size=32, verbose=0, class_weight=class_weight)
results_cnn_lstm_c5 = cross_validate(cnn_lstm_estimator_c5, data, label_c5, cv=kfold, scoring= scoring)
print(results_cnn_lstm_c5)
print('')

for st,vals in results_cnn_lstm_c5.items():
    print("Average_c5 for {} is {}".format(st,mean(vals)))
print('') 

#c6
class_weight = {0: 1.0/934.0,
                1: 1.0/274.0}
cnn_lstm_estimator_c6 = KerasClassifier(build_fn=cnn_lstm_model, epochs=10, batch_size=32, verbose=0, class_weight=class_weight)
results_cnn_lstm_c6 = cross_validate(cnn_lstm_estimator_c6, data, label_c6, cv=kfold, scoring= scoring)
print(results_cnn_lstm_c6)
print('')

for st,vals in results_cnn_lstm_c6.items():
    print("Average_c6 for {} is {}".format(st,mean(vals)))
print('') 

#c7

class_weight = {0: 1.0/948.0,
                1: 1.0/260.0}

cnn_lstm_estimator_c7 = KerasClassifier(build_fn=cnn_lstm_model, epochs=10, batch_size=32, verbose=0, class_weight=class_weight)
results_cnn_lstm_c7 = cross_validate(cnn_lstm_estimator_c7, data, label_c7, cv=kfold, scoring= scoring)
print(results_cnn_lstm_c7)
print('')

for st,vals in results_cnn_lstm_c7.items():
    print("Average_c7 for {} is {}".format(st,mean(vals)))
print('') 

# visualizing cnn_lstm model
model = Sequential()
model.add(Embedding(max_words, embedding_dim,input_length=maxlen))
model.add(Dropout(0.2))
model.add(Conv1D(filters,kernel_size,padding='valid',activation='relu',strides=1))
model.add(MaxPooling1D())
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.summary()

#lstm_cnn
from keras.models import Sequential
from keras.layers import Dropout, Conv1D, MaxPooling1D, LSTM, Dense, Embedding, Flatten
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

import warnings
from scipy import mean
warnings.filterwarnings(action = 'once')

# set parameters:
filters = 250
kernel_size = 3

seed = 7

def lstm_cnn_model():
    model = Sequential()
    model.add(Embedding(max_words, embedding_dim,input_length=maxlen))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    model.add(Conv1D(filters,kernel_size,padding='valid',activation='relu',strides=1))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.layers[0].set_weights([embedding_matrix])
    #freezing the embedding layer form updating
    model.layers[0].trainable = False
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
    return model

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_micro': 'recall_macro'}

#c3
class_weight = {0: 1.0/616.0,
                1: 1.0/592.0}
lstm_cnn_estimator_c3 = KerasClassifier(build_fn=lstm_cnn_model, epochs=10, batch_size=32, verbose=0, class_weight=class_weight)
results_lstm_cnn_c3 = cross_validate(lstm_cnn_estimator_c3, data, label_c3, cv=kfold, scoring= scoring)
print(results_lstm_cnn_c3)
print('')

for st,vals in results_lstm_cnn_c3.items():
    print("Average_c3 for {} is {}".format(st,mean(vals)))
print('') 

#c5
class_weight = {0: 1.0/1004.0,
                1: 1.0/204.0}
lstm_cnn_estimator_c5 = KerasClassifier(build_fn=lstm_cnn_model, epochs=10, batch_size=32, verbose=0, class_weight=class_weight)
results_lstm_cnn_c5 = cross_validate(lstm_cnn_estimator_c5, data, label_c5, cv=kfold, scoring= scoring)
print(results_lstm_cnn_c5)
print('')

for st,vals in results_lstm_cnn_c5.items():
    print("Average_c5 for {} is {}".format(st,mean(vals)))
print('') 

#c6
class_weight = {0: 1.0/934.0,
                1: 1.0/274.0}
lstm_cnn_estimator_c6 = KerasClassifier(build_fn=lstm_cnn_model, epochs=10, batch_size=32, verbose=0, class_weight=class_weight)
results_lstm_cnn_c6 = cross_validate(lstm_cnn_estimator_c6, data, label_c6, cv=kfold, scoring= scoring)
print(results_lstm_cnn_c6)
print('')

for st,vals in results_lstm_cnn_c6.items():
    print("Average_c6 for {} is {}".format(st,mean(vals)))
print('') 

#c7

class_weight = {0: 1.0/948.0,
                1: 1.0/260.0}

lstm_cnn_estimator_c7 = KerasClassifier(build_fn=lstm_cnn_model, epochs=10, batch_size=32, verbose=0, class_weight=class_weight)
results_lstm_cnn_c7 = cross_validate(lstm_cnn_estimator_c7, data, label_c7, cv=kfold, scoring= scoring)
print(results_lstm_cnn_c7)
print('')

for st,vals in results_lstm_cnn_c7.items():
    print("Average_c7 for {} is {}".format(st,mean(vals)))
print('') 

#visualizing lstm_cnn model
model = Sequential()
model.add(Embedding(max_words, embedding_dim,input_length=maxlen))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
model.add(Conv1D(filters,kernel_size,padding='valid',activation='relu',strides=1))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.summary()


# create plots to compare model performance C3

import matplotlib.pyplot as plt
import numpy as np

all_acc = pd.concat(
[pd.DataFrame(results_conv1d_c3['test_acc']),
pd.DataFrame(results_lstm_c3['test_acc']),
pd.DataFrame(results_lstm_mp_c3['test_acc']),
pd.DataFrame(results_cnn_lstm_c3['test_acc']),
pd.DataFrame(results_lstm_cnn_c3['test_acc'])], axis=1)

all_acc.columns = ['CNN', 'LSTM', 'LSTM+MeanPooling', 'CNN+LSTM', 'LSTM+CNN']

all_rec = pd.concat(
[pd.DataFrame(results_conv1d_c3['test_rec_micro']),
pd.DataFrame(results_lstm_c3['test_rec_micro']),
pd.DataFrame(results_lstm_mp_c3['test_rec_micro']),
pd.DataFrame(results_cnn_lstm_c3['test_rec_micro']),
pd.DataFrame(results_lstm_cnn_c3['test_rec_micro'])], axis=1)

all_prec = pd.concat(
[pd.DataFrame(results_conv1d_c3['test_prec_macro']),
pd.DataFrame(results_lstm_c3['test_prec_macro']),
pd.DataFrame(results_lstm_mp_c3['test_prec_macro']),
pd.DataFrame(results_cnn_lstm_c3['test_prec_macro']),
pd.DataFrame(results_lstm_cnn_c3['test_prec_macro'])], axis=1)

fig.clf()

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30,10), sharey=True)

axes[0].boxplot(all_acc.T, showmeans=True, vert=True, labels=all_acc.columns)
axes[0].set_title('Accuracy', fontsize=25)

axes[1].boxplot(all_rec.T, showmeans=True, vert=True, labels=all_acc.columns)
axes[1].set_title('Recall', fontsize=25)

axes[2].boxplot(all_prec.T, showmeans=True, vert=True, labels=all_acc.columns)
axes[2].set_title('Precision', fontsize=25)

fig.subplots_adjust(wspace=0.1)

for ax in fig.axes:
    matplotlib.pyplot.sca(ax)
    plt.xticks(rotation=90)
    
fig.suptitle('Models performance comparison for label C3', fontsize=30)

plt.show()

# create plots to compare model performance C5

all_acc = pd.concat(
[pd.DataFrame(results_conv1d_c5['test_acc']),
pd.DataFrame(results_lstm_c5['test_acc']),
pd.DataFrame(results_lstm_mp_c5['test_acc']),
pd.DataFrame(results_cnn_lstm_c5['test_acc']),
pd.DataFrame(results_lstm_cnn_c5['test_acc'])], axis=1)

all_acc.columns = ['CNN', 'LSTM', 'LSTM+MeanPooling', 'CNN+LSTM', 'LSTM+CNN']

all_rec = pd.concat(
[pd.DataFrame(results_conv1d_c5['test_rec_micro']),
pd.DataFrame(results_lstm_c5['test_rec_micro']),
pd.DataFrame(results_lstm_mp_c5['test_rec_micro']),
pd.DataFrame(results_cnn_lstm_c5['test_rec_micro']),
pd.DataFrame(results_lstm_cnn_c5['test_rec_micro'])], axis=1)

all_prec = pd.concat(
[pd.DataFrame(results_conv1d_c5['test_prec_macro']),
pd.DataFrame(results_lstm_c5['test_prec_macro']),
pd.DataFrame(results_lstm_mp_c5['test_prec_macro']),
pd.DataFrame(results_cnn_lstm_c5['test_prec_macro']),
pd.DataFrame(results_lstm_cnn_c5['test_prec_macro'])], axis=1)

fig.clf()

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30,10), sharey=True)

axes[0].boxplot(all_acc.T, showmeans=True, vert=True, labels=all_acc.columns)
axes[0].set_title('Accuracy', fontsize=25)

axes[1].boxplot(all_rec.T, showmeans=True, vert=True, labels=all_acc.columns)
axes[1].set_title('Recall', fontsize=25)

axes[2].boxplot(all_prec.T, showmeans=True, vert=True, labels=all_acc.columns)
axes[2].set_title('Precision', fontsize=25)

fig.subplots_adjust(wspace=0.1)

for ax in fig.axes:
    matplotlib.pyplot.sca(ax)
    plt.xticks(rotation=90)
    
fig.suptitle('Models performance comparison for label C5', fontsize=30)

plt.show()

# create plots to compare model performance C6
all_acc = pd.concat(
[pd.DataFrame(results_conv1d_c6['test_acc']),
pd.DataFrame(results_lstm_c6['test_acc']),
pd.DataFrame(results_lstm_mp_c6['test_acc']),
pd.DataFrame(results_cnn_lstm_c6['test_acc']),
pd.DataFrame(results_lstm_cnn_c6['test_acc'])], axis=1)

all_acc.columns = ['CNN', 'LSTM', 'LSTM+MeanPooling', 'CNN+LSTM', 'LSTM+CNN']

all_rec = pd.concat(
[pd.DataFrame(results_conv1d_c6['test_rec_micro']),
pd.DataFrame(results_lstm_c6['test_rec_micro']),
pd.DataFrame(results_lstm_mp_c6['test_rec_micro']),
pd.DataFrame(results_cnn_lstm_c6['test_rec_micro']),
pd.DataFrame(results_lstm_cnn_c6['test_rec_micro'])], axis=1)

all_prec = pd.concat(
[pd.DataFrame(results_conv1d_c6['test_prec_macro']),
pd.DataFrame(results_lstm_c6['test_prec_macro']),
pd.DataFrame(results_lstm_mp_c6['test_prec_macro']),
pd.DataFrame(results_cnn_lstm_c6['test_prec_macro']),
pd.DataFrame(results_lstm_cnn_c6['test_prec_macro'])], axis=1)

fig.clf()

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30,10), sharey=True)

axes[0].boxplot(all_acc.T, showmeans=True, vert=True, labels=all_acc.columns)
axes[0].set_title('Accuracy', fontsize=25)

axes[1].boxplot(all_rec.T, showmeans=True, vert=True, labels=all_acc.columns)
axes[1].set_title('Recall', fontsize=25)

axes[2].boxplot(all_prec.T, showmeans=True, vert=True, labels=all_acc.columns)
axes[2].set_title('Precision', fontsize=25)

fig.subplots_adjust(wspace=0.1)

for ax in fig.axes:
    matplotlib.pyplot.sca(ax)
    plt.xticks(rotation=90)
    
fig.suptitle('Models performance comparison for label C6', fontsize=30)

plt.show()


# create plots to compare model performance C7

all_acc = pd.concat(
[pd.DataFrame(results_conv1d_c7['test_acc']),
pd.DataFrame(results_lstm_c7['test_acc']),
pd.DataFrame(results_lstm_mp_c7['test_acc']),
pd.DataFrame(results_cnn_lstm_c7['test_acc']),
pd.DataFrame(results_lstm_cnn_c7['test_acc'])], axis=1)

all_acc.columns = ['CNN', 'LSTM', 'LSTM+MeanPooling', 'CNN+LSTM', 'LSTM+CNN']

all_rec = pd.concat(
[pd.DataFrame(results_conv1d_c7['test_rec_micro']),
pd.DataFrame(results_lstm_c7['test_rec_micro']),
pd.DataFrame(results_lstm_mp_c7['test_rec_micro']),
pd.DataFrame(results_cnn_lstm_c7['test_rec_micro']),
pd.DataFrame(results_lstm_cnn_c7['test_rec_micro'])], axis=1)

all_prec = pd.concat(
[pd.DataFrame(results_conv1d_c7['test_prec_macro']),
pd.DataFrame(results_lstm_c7['test_prec_macro']),
pd.DataFrame(results_lstm_mp_c7['test_prec_macro']),
pd.DataFrame(results_cnn_lstm_c7['test_prec_macro']),
pd.DataFrame(results_lstm_cnn_c7['test_prec_macro'])], axis=1)


fig.clf()

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30,10), sharey=True)

axes[0].boxplot(all_acc.T, showmeans=True, vert=True, labels=all_acc.columns)
axes[0].set_title('Accuracy', fontsize=25)

axes[1].boxplot(all_rec.T, showmeans=True, vert=True, labels=all_acc.columns)
axes[1].set_title('Recall', fontsize=25)

axes[2].boxplot(all_prec.T, showmeans=True, vert=True, labels=all_acc.columns)
axes[2].set_title('Precision', fontsize=25)

fig.subplots_adjust(wspace=0.1)

for ax in fig.axes:
    matplotlib.pyplot.sca(ax)
    plt.xticks(rotation=90)
    
fig.suptitle('Models performance comparison for label C7', fontsize=30)

plt.show()



# compare run time for all models

train_time = pd.concat(
[pd.DataFrame(results_conv1d_c3['fit_time']),
pd.DataFrame(results_lstm_c3['fit_time']),
pd.DataFrame(results_lstm_mp_c3['fit_time']),
pd.DataFrame(results_cnn_lstm_c3['fit_time']),
pd.DataFrame(results_lstm_cnn_c3['fit_time'])], axis=1)

train_time.columns = ['CNN', 'LSTM', 'LSTM+MeanPooling', 'CNN+LSTM', 'LSTM+CNN']

plt.figure()

train_time.mean().plot(kind='bar', width=0.7)

plt.ylabel("Fit Time", fontsize=20)
plt.title("Fit Time Comparison for all models", fontsize=20)
plt.show()


# prediction for the selected model : CNN
# prepare the test set for prediction
tokenizer_test = Tokenizer(num_words=max_words)
tokenizer_test.fit_on_texts(test_posts)
sequences_test = tokenizer_test.texts_to_sequences(test_posts)
word_index_test = tokenizer_tets.word_index
data_test = pad_sequences(sequences_test, maxlen=maxlen)

label_test_c3 = np.array(test_labels['c3'])
label_test_c5 = np.array(test_labels['c5'])
label_test_c6 = np.array(test_labels['c6'])
label_test_c7 = np.array(test_labels['c7'])


from sklearn.model_selection import cross_val_predict
y_pred_3 = cross_val_predict(conv1d_estimator_c3, data_test, label_test_c3, cv=kfold)
y_pred_5 = cross_val_predict(conv1d_estimator_c5, data_test, label_test_c5, cv=kfold)
y_pred_6 = cross_val_predict(conv1d_estimator_c6, data_test, label_test_c6, cv=kfold)
y_pred_7 = cross_val_predict(conv1d_estimator_c7, data_test, label_test_c7, cv=kfold)

