import re
import matplotlib.pyplot as plt
import pandas as pd
import spacy

spacy_nlp = spacy.load('en_core_web_sm')
all_stopwords = spacy_nlp.Defaults.stop_words

def format_tweet_data_frame(root_path, tweet_files):
    tweets = []
    for filename in tweet_files:
        df = pd.read_csv(f"{root_path}/{filename}",  delimiter = "\t", header=None)
        tweets.append(df)
    
    frame = pd.concat(tweets, axis=0, ignore_index=True)
    frame['processed_tweet'] = frame[2].apply(clean_and_tokenize)
    return frame

def plot_sentiment_distribution(pos_count, neutral_count, neg_count, title):
    fig = plt.figure(figsize=(5, 5))
    labels = 'Positive', 'Neutral', 'Negative'
    sizes = [pos_count, neutral_count, neg_count] 
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
    plt.axis('equal')  
    plt.title(title)
    plt.show()

def clean_tweet(tweet):
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)

    # remove hyperlinks
    # tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    tweet = re.sub(r'http\S+', '', tweet)

    # # remove hashtags
    # # only removing the hash tags # sign from the word
    tweet = re.sub(r'#', '', tweet)

    # # remove usernames
    tweet = re.sub('@[\w]+','',tweet)
    return tweet

def tokenize_and_remove_stop_words(tweet):
    doc = spacy_nlp(tweet)
    tokens = [token.text.lower() for token in doc if not (token.is_stop or token.is_punct or str.isspace(token.text))]
    return " ".join(tokens)

def clean_and_tokenize(tweet):
  return tokenize_and_remove_stop_words(clean_tweet(tweet))
