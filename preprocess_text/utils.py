############################
# Import Libraries/Modules #
############################

# System Libraries
import os
import sys

# Data Manipulation
import pandas as pd
import numpy as np

# Text Preprocessing
import spacy
from spacy.lang.en.stop_words import STOP_WORDS as stopwords
from bs4 import BeautifulSoup
import unicodedata
from textblob import TextBlob

nlp = spacy.load('en_core_web_sm')

# Text Cleaning - Emoji & Emoticons
import re
import pickle
from emot.emo_unicode import UNICODE_EMOJI, EMOTICONS_EMO



#############
# Functions #
#############

# Note: Putting underscore to signify that these are private and internal methods
# Function to count word
def _get_wordcounts(x):
    length = len(str(x).split())
    return length


# Function to count character
def _get_char_counts(x):
    s = x.split()
    x = ''.join(s)
    return len(x)


# Function to calculate average wordlength
def _get_avg_wordlength(x):
    return _get_char_counts(x)/_get_wordcounts(x)


# Function to count stopword
def _get_stopwords_counts(x):
    return len([t for t in x.split() if t in stopwords])


# Function to count punctuation
def _get_punc_counts(x):
    punc = re.findall(r'[^\w ]+', x)
    counts = len(punc)
    return counts


# Function to count hashtag
def _get_hashtag_counts(x):
    return len([t for t in x.split() if t.startswith('#')])


# Function to count mentions
def _get_mention_counts(x):
    return len([t for t in x.split() if t.startswith('@')])


# Function to count digit/numeric
def _get_digit_counts(x):
    return len([t for t in x.split() if t.isdigit()])


# Function to count uppercase
def _get_uppercase_counts(x):
    return len([t for t in x.split() if t.isupper()])


# Function to expand all contractions
def _cont_exp(x):
    contractions = { 
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how does",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    " u ": " you ",
    " ur ": " your ",
    " n ": " and ",
    "won't": "would not",
    'dis': 'this',
    'bak': 'back',
    'brng': 'bring'}

    if type(x) is str:
        for key in contractions:
            value = contractions[key]
            x = x.replace(key, value)
        return x
    else:
        return x


# Function for counting occurence of emails
def _get_emails(x):
    emails = re.findall(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+\b)', x)
    counts = len(emails)
    return counts, emails


# Function for removing emails
def _remove_emails(x):
    return re.sub(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)',"", x)


# Function for counting occurence of weblink
def _get_urls(x):
    urls = re.findall(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', x)
    counts = len(urls)
    return counts, urls


# Function for removing weblink
def _remove_urls(x):
    return re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', 'URL' , x)


# Function for removing mentions
def _remove_mention(x):
    return re.sub(r'(@)([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])', 'ALT_USER', x)


# Function for removing special characters/punctuation
def _remove_special_chars(x):
    x = re.sub(r'[^\w ]+', "", x)
    x = ' '.join(x.split())
    return x


# Function for removing elongated chars and reduction
def _remove_elongated_chars(x):
    return re.sub(r'(.)\1{2,}',r'\1',x) #any characters, numbers, symbols


# Function for removing HTML elements
def _remove_html_tags(x):
    return BeautifulSoup(x, 'lxml').get_text().strip()


# Function for removing accented character
def _remove_accented_chars(x):
    x = unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return x


# Function for removing numeric
def _remove_numeric(x):
    return ''.join([i for i in x if not i.isdigit()])


# Function for removing stop word
def _remove_stopwords(x):
    return ' '.join([t for t in x.split() if t not in stopwords])


# Function for making the word to base form
def _make_base(x):
    x = str(x)
    x_list = []
    doc = nlp(x)
    
    for token in doc:
        lemma = token.lemma_
        if lemma == '-PRON-' or lemma == 'be':
            lemma = token.text

        x_list.append(lemma)
    return ' '.join(x_list)



def _get_value_counts(df,col):
    text = ' '.join(df[col])
    text = text.split()
    freq = pd.Series(text).value_counts()
    return freq


# Function for removing common word
def _remove_common_words(x, freq, n=20):
    fn = freq[:n]
    x = ' '.join([t for t in x.split() if t not in fn])
    return x


# Function for removing rare word
def _remove_rarewords(x, freq, n=20):
    fn = freq.tail(20)
    x=' '.join([t for t in x.split() if t not in fn])
    return x


# Function for spelling correction
def _spelling_correction(x):
    x = TextBlob(x).correct()
    return x


# Function for counting emoticons 
pattern_emoticon = u'|'.join(k.replace('|','\\|') for k in EMOTICONS_EMO)
pattern_emoticon = pattern_emoticon.replace('\\','\\\\')
pattern_emoticon = pattern_emoticon.replace('(','\\(')
pattern_emoticon = pattern_emoticon.replace(')','\\)')
pattern_emoticon = pattern_emoticon.replace('[','\\[')
pattern_emoticon = pattern_emoticon.replace(']','\\]')
pattern_emoticon = pattern_emoticon.replace('*','\\*')
pattern_emoticon = pattern_emoticon.replace('+','\\+')
pattern_emoticon = pattern_emoticon.replace('^','\\^')
pattern_emoticon = pattern_emoticon.replace('·','\\·')
pattern_emoticon = pattern_emoticon.replace('\{','\\{·')

def _get_emoticon_counts(x):
    emoticon_pattern = re.compile(u'(' + pattern_emoticon + u')')
    emoticon = [i for i in emoticon_pattern.findall(x) if i]
    return len(emoticon)


# Function for converting emoticons into word
def _convert_emoticons(x):
    for emot in EMOTICONS_EMO:
        x = x.replace(emot, "_".join(EMOTICONS_EMO[emot].replace(",","").replace(":","").split()))
    return x


# Function for counting emoji
pattern_emoji = u'|'.join(k.replace('|','\\|') for k in UNICODE_EMOJI)
pattern_emoji = pattern_emoji.replace('\\','\\\\')
pattern_emoji = pattern_emoji.replace('(','\\(')
pattern_emoji = pattern_emoji.replace(')','\\)')
pattern_emoji = pattern_emoji.replace('[','\\[')
pattern_emoji = pattern_emoji.replace(']','\\]')
pattern_emoji = pattern_emoji.replace('*','\\*')
pattern_emoji = pattern_emoji.replace('+','\\+')
pattern_emoji = pattern_emoji.replace('^','\\^')
pattern_emoji = pattern_emoji.replace('·','\\·')
pattern_emoji = pattern_emoji.replace('\{','\\{·')
pattern_emoji = pattern_emoji.replace('\}','\\}·')

def _get_emoji_counts(x):
    emoji_pattern = re.compile(u'(' + pattern_emoji + u')')
    emoji = [i for i in emoji_pattern.findall(x) if i]
    return len(emoji)


# Function for converting emoji into word
def _convert_emojis(x):
    for emot in UNICODE_EMOJI:
        x = x.replace(emot, "_".join(UNICODE_EMOJI[emot].replace(",","").replace(":","").split()))
    return x