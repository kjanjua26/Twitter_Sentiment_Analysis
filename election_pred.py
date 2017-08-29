'''
	S1: Scrap off the tweets from Twitter related to results.
	S2: Scatter Graph the tweets based on polarity.
	S3: Keras model to predict the emotions.
	S4: Check on actual speech using conversion to text using cmu spinhx4.

	Dependencies: 
		
		Matplotlib
		Tweepy
		Numpy
		Scikit-Learn
		Keras

	Output: 
		Election Results and Emotions about Nawaz Sharif

'''
import matplotlib.pyplot as plt 
import keras 
from keras.layers import Dense, Activation, Embedding, LSTM
from keras.models import Sequential, load_model
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np 
import sklearn 
import tweepy as tp 
from textblob import TextBlob
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical


consumer_key = 'pKnvO75ckS2QAj7mSq8nnlMBv'
consumer_secret = 'xiZ4835Rp85NZTxMZEMux2PE69pKX9nYHiFogq32yCyoiW3GNE'
access_token = '709712385246961664-yI4TWDMYDJP0J8ERUgO4KzmKgCzSYnA'
access_token_secret = 'AX1aHYqzv4xR6KPzfqfCHnaqvrA5A7IsPFi82fMxCjwxM'

tweets = []
dates = []
neg_count = 0
pos_count = 0
t_sum = 0

data = pd.read_csv('twitter.csv')
max_features = 2000
tokenizer = Tokenizer(nb_words=max_features, split=' ')
tokenizer.fit_on_texts(data['text'].values)
x = tokenizer.texts_to_sequences(data['text'].values)
x = pad_sequences(x)

def model():
	model = Sequential()
	model.add(Embedding(max_features, 128))
	model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
	model.add(Dense(2, activation='softmax'))
	return model

model = model()
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())
y = pd.get_dummies(data['class']).values
X_train, x_test, Y_train, y_test = train_test_split(x,y, test_size = 0.33, random_state = 42)
batch_size = 32
model.fit(X_train, Y_train, nb_epoch = 10, batch_size=batch_size, verbose = 1)

validation_size = 50
x_valid = x_test[-validation_size:]
y_valid = y_test[-validation_size:]
x_test = x_test[:-validation_size]
y_test = y_test[:-validation_size]
score,acc = model.evaluate(x_test, y_test, verbose = 0, batch_size = batch_size)
print "\n"
print("The accuracy is: %.5f" % (acc))
print "\nSaving Model."

model.save('weights.h5')

def twitter_conn(consumer_key,consumer_secret,access_token,access_token_secret):
	auth = tp.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token,access_token_secret)
	api = tp.API(auth)
	return api

def search_tweets(api,keyword):
	ptweets = api.search(keyword)
	for tweet in ptweets:
		tweets.append(tweet.text)
		dates.append(tweet.created_at)
		return (tweets,dates)

api = twitter_conn(consumer_key,consumer_secret,access_token,access_token_secret)
t, d = search_tweets(api, 'love')

model = load_model('weights.h5')
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
max_features = 2000

#t is a list and can be done using for loop.
#1 means postivie, 0 means negative
for i in t:
	tokenizer = Tokenizer(nb_words=max_features, split=' ')
	tokenizer.fit_on_texts(i)
	x = tokenizer.texts_to_sequences(i)
	x = pad_sequences(x)
	out = model.predict_classes(x,batch_size=32)
	for j in out:
		if j == 0:
			global neg_count
			global t_sum
			print "The sentiment is Negative."
			neg_count += 1
			t_sum += 1
		if j == 1:
			global t_sum
			global pos_count
			pos_count += 1
			t_sum += 1
			print "The sentiment is Postive."

def plot(t_sum,neg,pos):
	fig = plt.gcf()
	fig.canvas.set_window_title('Twitter Analysis')
	plt.xlim(0,200)
	plt.scatter(t_sum,neg,color='black', label='Negative')
	plt.scatter(t_sum,pos,color='red', label='Positive')
	plt.ylabel('Positive/Negative')
	plt.xlabel('Tweet Count')
	plt.title('Love')
	plt.legend()
	plt.show()
plot(t_sum,neg_count,pos_count)
