import tweepy
from textblob import TextBlob
import csv

# Step 1 - Authenticate
consumer_key= '6c0wdESF7bPZlcKAnfJymKOmU'
consumer_secret= 'GYKnEO4XSQeOUbbYxM6o9rB4oUKMyE4mXK53G7bcCKOogKirEn'

access_token='2549467563-FIwbhLlsfCMqcjNDmjpk7cEUIg9UrWMtqmEFcCx'
access_token_secret='fBv2ZTbUx5nAKhqFMqzSGIaqVTG5quEv3EAeQa5HIjWhp'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

#Step 3 - Retrieve Tweets
public_tweets = api.search('Trump')



#CHALLENGE - Instead of printing out each tweet, save each Tweet to a CSV file
#and label each one as either 'positive' or 'negative', depending on the sentiment 
#You can decide the sentiment polarity threshold yourself




with open('tweets.csv', 'w', newline='') as csvfile:
	spamwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)

	for tweet in public_tweets:
		analysis = TextBlob(tweet.text)
		if analysis.sentiment.polarity > 0:
			result = 'positive'
		else: result = 'negative'

		spamwriter.writerow([tweet.text] + [result])
