# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 13:53:18 2020

@author: user
"""

import pandas as pd 
import numpy as np
import re
textdata = pd.read_csv('HW4_train.csv',sep="\t")
p = re.compile(r'[^\w\s]+')
idx = np.flatnonzero(textdata['text'].notna()) #索引 nonmissing value 做為行
col_idx = textdata.columns.get_loc('text') #get position of columns 作為列
textdata.iloc[idx,col_idx] = [p.sub('', x) for x in textdata.iloc[idx,col_idx].tolist()]

regex = re.compile(r'[^a-zA-Z$ ]')
#regex = re.compile(r'^[a-zA-Z_]*$')
textdata['text'] = [regex.sub('', x) for x in textdata['text'].tolist()]
textdata['text'].str.lower()
from sklearn.feature_extraction.text import CountVectorizer
stop_words=['english']
cv = CountVectorizer(stop_words=stop_words,max_features=5000,decode_error="replace",
                     analyzer='word',
                     token_pattern=r'\w{1,}')
x_train = cv.fit_transform(textdata['text'])
cv.get_feature_names()
x_train = pd.DataFrame(x_train.toarray(), columns=cv.get_feature_names())

testdata =pd.read_csv('test.csv',sep="\t")
p = re.compile(r'[^\w\s]+')
idx = np.flatnonzero(testdata['text'].notna()) #索引 nonmissing value 做為行
col_idx = testdata.columns.get_loc('text') #get position of columns 作為列
testdata.iloc[idx,col_idx] = [p.sub('', x) for x in testdata.iloc[idx,col_idx].tolist()]
regex = re.compile(r'[^a-zA-Z$ ]')
testdata['text'] = [regex.sub('', x) for x in testdata['text'].tolist()]
testdata['text'].str.lower()
stop_words=['english']
cv = CountVectorizer(stop_words=stop_words,max_features=5000,decode_error="replace",
                     analyzer='word',
                     token_pattern=r'\w{1,}')
y_train = cv.fit_transform(testdata['text'])
cv.get_feature_names()
y_train = pd.DataFrame(y_train.toarray(), columns=cv.get_feature_names())

from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(smooth_idf=True)
train_features = transformer.fit_transform(x_train)
test_features =  transformer.fit_transform(y_train)
train_label =textdata['label']




from xgboost  import XGBClassifier
model = XGBClassifier(n_estimators=1000,learning_rate= 0.3,seed=1000,max_depth=8)
model.fit(train_features, train_label)
y_pred = model.predict(test_features)
y_pred= y_pred.astype(int)

sample=pd.read_csv('sample_submission.csv')
sample_label=sample['label']

from sklearn.metrics import confusion_matrix
from sklearn import metrics

accuracy = metrics.accuracy_score(sample_label, y_pred)
recall = metrics.recall_score(sample_label, y_pred)
precision = metrics.precision_score(sample_label, y_pred)
f1 = metrics.f1_score(sample_label, y_pred)
    print('xgboost Model accuracy:',accuracy)
    print('xgboost Model recall:',recall)
    print('xgboost Model precision:',precision)
    print('xgboost Model f1:',f1)

pip install lightgbm
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
LGBMClassifier(n_estimators=90, random_state =94, max_depth=5,num_leaves=31,objective='binary')

clf = lgb.LGBMClassifier()
clf.fit(train_features, train_label)
y_pred2=clf.predict(test_features)
y_pred2= y_pred2.astype(int)
from sklearn.metrics import accuracy_score
accuracy=accuracy_score( sample_label,y_pred2)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(sample_label,y_pred2)
print('LightGBM Model Confusion matrix\n\n', cm)
precision = metrics.precision_score(sample_label,y_pred2)
recall = metrics.recall_score(sample_label, y_pred2)
f1 = metrics.f1_score(sample_label, y_pred2)
print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(sample_label, y_pred2)))
print('LightGBM Model precision score: {0:0.4f}',precision)
print('LightGBM Model recall score: {0:0.4f}',recall)
print('LightGBM Model f1 score: {0:0.4f}',f1)
###################################################
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(n_estimators=20, learning_rate = 0.5, max_features=800, max_depth = 2, random_state = 0)
gb.fit(train_features, train_label)
predictions = gb.predict(test_features)
y_pred3= predictions.astype(int)

cm = confusion_matrix(sample_label,y_pred3)
print('GBDT Model Confusion matrix\n\n', cm)
accuracy=accuracy_score( sample_label,y_pred3)
precision = metrics.precision_score(sample_label,y_pred3)
recall = metrics.recall_score(sample_label, y_pred3)
f1 = metrics.f1_score(sample_label, y_pred2)
print('GBDT Model accuracy score: {0:0.4f}'.format(accuracy_score(sample_label, y_pred3)))
print('GBDT Model precision score: {0:0.4f}',precision)
print('GBDT Model recall score: {0:0.4f}',recall)
print('LightGBM Model f1 score: {0:0.4f}',f1)



