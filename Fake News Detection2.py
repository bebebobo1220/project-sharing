# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 14:57:27 2020

@author: user
"""
import pandas as pd 
import numpy as np
import re
textdata = pd.read_csv('HW4_train.csv',sep="\t")
testdata =pd.read_csv('test.csv',sep="\t")
p = re.compile(r'[^\w\s]+')
idx = np.flatnonzero(textdata['text'].notna())
col_idx = textdata.columns.get_loc('text')
textdata.iloc[idx,col_idx] = [p.sub('', x) for x in textdata.iloc[idx,col_idx].tolist()]
regex = re.compile(r'[^a-zA-Z$ ]')
textdata['text'] = [regex.sub('', x) for x in textdata['text'].tolist()]
textdata["text"]=textdata['text'].str.lower()

idx2 = np.flatnonzero(testdata['text'].notna()) #索引 nonmissing value 做為行
col_idx2 = testdata.columns.get_loc('text') #get position of columns 作為列
testdata.iloc[idx2,col_idx2] = [p.sub('', x) for x in testdata.iloc[idx2,col_idx2].tolist()]
testdata['text'] = [regex.sub('', x) for x in testdata['text'].tolist()]
testdata['text']=testdata['text'].str.lower()

X= textdata["text"]
y =textdata["label"]
############################################################
##countvectorizer
from sklearn.feature_extraction.text import CountVectorizer
stop_words=['english']
cv1 = CountVectorizer(stop_words=stop_words,max_features=380,decode_error="replace",
                     analyzer='word',
                     token_pattern=r'\w{1,}')
train= cv1.fit_transform(textdata["text"])
cv1.get_feature_names()
train = pd.DataFrame(train.toarray(), columns=cv1.get_feature_names())
cv2 = CountVectorizer(stop_words=stop_words,max_features=380,decode_error="replace",
                     analyzer='word',
                     token_pattern=r'\w{1,}')

test= cv2.fit_transform(testdata['text'])
cv2.get_feature_names()
test= pd.DataFrame(test.toarray(), columns=cv2.get_feature_names())

##tf idf
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(smooth_idf=True)
train_features = transformer.fit_transform(train)
test_features =  transformer.fit_transform(test)

#####################################################################
##model pre-processing
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(y)
Y = Y.reshape(-1,1)

#from sklearn.model_selection import train_test_split
#X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
max_words = 3000
max_len = 150
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X)
sequences = tok.texts_to_sequences(X)
sequences_matrix = pad_sequences(sequences,maxlen=max_len)

from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,32,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model
model = RNN()
model.summary()
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(sequences_matrix,Y,batch_size=128,epochs=10,
          validation_split=0.2)

x_test = testdata["text"]

sample=pd.read_csv('sample_submission.csv')
y2=sample['label']
Y2= le.fit_transform(y2)
y_test= Y2.reshape(-1,1)

test_sequences = tok.texts_to_sequences(x_test)
test_sequences_matrix = pad_sequences(test_sequences,maxlen=max_len)

accr = model.evaluate(test_sequences_matrix,y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
#############################################################################

from keras.layers.recurrent import SimpleRNN


def simpleRNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,32,input_length=max_len)(inputs)
    layer = SimpleRNN(16)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.35)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model2 = Model(inputs=inputs,outputs=layer)
    return model2

model2 = simpleRNN()
model2.summary()
model2.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model2.fit(sequences_matrix,Y,batch_size=128,epochs=10,validation_split=0.2)

accr2 = model2.evaluate(test_sequences_matrix,y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr2[0],accr2[1]))





###############################################################################













