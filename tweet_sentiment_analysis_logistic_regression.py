# sentiment analysis using NLTK Twitter dataset (twitter_samples)

import re
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

#______________________________________________________

from nltk.corpus import twitter_samples  

#nltk.download('twitter_samples')  #It contains 5000 positive tweets and 5000 negative tweets 
#nltk.download('stopwords')

pos_tweets = twitter_samples.strings('positive_tweets.json')  # a list of strings
neg_tweets = twitter_samples.strings('negative_tweets.json')

# spliting the data into traiing and testing
train_pos = pos_tweets[:4000]
train_neg = neg_tweets[:4000]
test_pos = pos_tweets[4000:]
test_neg = neg_tweets[4000:]


train_x = train_pos + train_neg 
test_x = test_pos + test_neg

train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)

#_______________________________________________________________________________________
# preprocessing Tweets using NLTK library

def process_tweet(tweet):

    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')

    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    # remove hashtags
    tweet = re.sub(r'#', '', tweet)
   
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and
                word not in string.punctuation):
            stem_word = stemmer.stem(word)
            tweets_clean.append(stem_word)

    return tweets_clean

#_________________________________________________________________
# Building a dictionary where we can look up how many times a word appears in the list of positive or negative tweets

def build_freqs(tweets, ys):
    
    yslist = np.squeeze(ys).tolist()
    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1

    return freqs

freqs = build_freqs(train_x, train_y)

#____________________________________________________________________

# creating the feature vectore of dimension (1,3)

def extract_features(tweet, freqs):


    words = process_tweet(tweet)
    x = np.zeros((1, 3)) 
    #bias term is set to 1
    x[0,0] = 1 

    for word in words:
        x[0,1] += freqs.get((word, 1),0)
        x[0,2] += freqs.get((word, 0),0)

    assert(x.shape == (1, 3))
    return x

#_________________________________________________________________
# creating the sigmoid function for logistic regression

def sigmoid(z): 

    h = 1 / (1 + np.exp(- z))
    
    return h

#___________________________________________________________________
# creating gradient descent function  for calculating cost and updating theta

def gradientDescent(x, y, theta, alpha, num_iters):

    m = np.shape(x)[0]
    
    for i in range(0, num_iters):
        
        z = np.dot(x,theta)
        h = sigmoid(z)
        J = - 1/m * ( np.dot (y.T,np.log(h)) + np.dot ((1-y).T,np.log(1-h)))
        theta = theta - (alpha/m* np.dot(x.T,(h-y)))

    J = float(J)
    return J, theta

# Training the model
# creating features matrix

X = np.zeros((len(train_x), 3))

for i in range(len(train_x)):
    X[i, :]= extract_features(train_x[i], freqs)

Y = train_y
J, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 1500)

print(J)
print(theta)

#______________________________________________________________
# Prediction

def predict_tweet(tweet, freqs, theta):

    x = extract_features(tweet, freqs)
    y_pred = sigmoid ( np.dot(x,theta) )
    
    return y_pred
#___________________________________________________________
# Accuracy 

def test_logistic_regression(test_x, test_y, freqs, theta):

    y_hat = []
    
    for tweet in test_x:

        y_pred = predict_tweet(tweet, freqs, theta)
        if y_pred > 0.5:
            y_hat.append(1)
        else:

            y_hat.append(0)

    y_hat = np.asarray(y_hat)
    
    accuracy = (np.squeeze(y_hat) == np.squeeze(test_y)).mean()

    return accuracy

accuracy = test_logistic_regression(test_x, test_y, freqs, theta)
print(accuracy)

#______________________________________________________________________

my_tweet = 'This weeks tribute is up on insta! ❤️👑'

y_hat = predict_tweet(my_tweet, freqs, theta)


if y_hat > 0.5:
    print('Positive sentiment')
else: 
    print('Negative sentiment')

