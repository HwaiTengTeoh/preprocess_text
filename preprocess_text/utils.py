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
import string
import pickle
from emot.emo_unicode import UNICODE_EMOJI, EMOTICONS_EMO



#############
# Functions #
#############

# Note: Putting underscore to signify that these are private and internal methods
# count word
def _get_wordcounts(x):
    length = len(str(x).split())
    return length


# Count character
def _get_char_counts(x):
    s = x.split()
    x = ''.join(s)
    return len(x)


# Calculate average wordlength
def _get_avg_wordlength(x):
    return _get_char_counts(x)/_get_wordcounts(x)


# Count stopword
def _get_stopwords_counts(x):
    return len([t for t in x.split() if t in stopwords])


# Count punctuation
def _get_punc_counts(x):
    punc = re.findall(r'[^\w ]+', x)
    counts = len(punc)
    return counts


# Count hashtag
def _get_hashtag_counts(x):
    return len([t for t in x.split() if t.startswith('#')])


# Count mentions
def _get_mention_counts(x):
    return len([t for t in x.split() if t.startswith('@')])


# Count digit/numeric
def _get_digit_counts(x):
    return len([t for t in x.split() if t.isdigit()])


# Count uppercase
def _get_uppercase_counts(x):
    return len([t for t in x.split() if t.isupper()])


# Expand all contractions
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


# Count occurence of emails
def _get_emails(x):
    emails = re.findall(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+\b)', x)
    counts = len(emails)
    return counts


# Remove emails
def _remove_emails(x):
    return re.sub(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)',"", x)


# Count occurence of weblink
def _get_urls(x):
    urls = re.findall(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', x)
    counts = len(urls)
    return counts


# Remove weblink
def _remove_urls(x):
    return re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '' , x)


# Remove mentions
def _remove_mention(x):
    return re.sub(r'(@)([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])', '', x)


# Remove special characters/punctuation
punct = string.punctuation
def _remove_special_chars(x):
    for p in punct:
        x = x.replace(p, " ")
    return x

# Remove elongated chars and reduction
def _remove_elongated_chars(x):
    temp = re.sub(r'(.)\1{2,}',r'\1',x)   #any characters, numbers, symbols
    temp2 = re.sub(r'(..)\1{2,}', r'\1', temp)  
    temp3 = re.sub(r'(...)\1{2,}', r'\1', temp2)
    temp4 = re.sub(r'(....)\1{2,}', r'\1', temp3)  
    return temp4

# Remove HTML elements
def _remove_html_tags(x):
    return BeautifulSoup(x, 'lxml').get_text().strip()


# Remove accented character
def _remove_accented_chars(x):
    x = unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return x


# Remove numeric
def _remove_numeric(x):
    return ''.join([i for i in x if not i.isdigit()])


# Remove stop word
# def _remove_stopwords(x):
#     return ' '.join([t for t in x.split() if t not in stopwords])

def _remove_stopwords(x, keep_pronoun=True):
    x = str(x)
    x_list = []
    doc = nlp(x)
    
    for token in doc:
        text = token.text
        
        if token.is_stop:
            if keep_pronoun & (token.pos_ == 'PRON' or token.pos_ == 'PROPN'):
                text = token.text
            else:
                text = ''
        else:
            text = token.text
        x_list.append(text.strip())
    return ' '.join(x_list)


# Convert the word to base form
def _make_base(x):
    x = str(x)
    x_list = []
    doc = nlp(x)
    
    for token in doc:
        lemma = token.lemma_
        
        if token.pos_ == 'PRON' or token.lemma_ == 'be':
            lemma = token.text

        x_list.append(lemma)
    return ' '.join(x_list)



def _get_value_counts(df,col):
    text = ' '.join(df[col])
    text = text.split()
    freq = pd.Series(text).value_counts()
    return freq


# Remove common word
def _remove_common_words(x, freq, n=20):
    fn = freq[:n]
    x = ' '.join([t for t in x.split() if t not in fn])
    return x


# Remove rare word
def _remove_rarewords(x, freq, n=20):
    fn = freq.tail(20)
    x=' '.join([t for t in x.split() if t not in fn])
    return x


# Spelling correction
def _spelling_correction(x):
    x = TextBlob(x).correct()
    return x


# Count emoticons 
# with open('EMOTICONS_EMO_ALL.pkl', 'rb') as fp:
#     EMOTICONS_EMO_ALL = pickle.load(fp)

add_emoticon = {'-.-': 'shame',
      '-_-': 'squiting',
      '^.^': 'happy face',
      ':0': 'surprise',
      '^-^': 'happy face',
      ':33': 'happy face smiley',
      '^__^': 'happy face',
      '-____-': 'shame',
      'o_o': 'confused',
      'O_O': 'confused',
      'x3': 'Cute face'
      }

EMOTICONS_EMO.update(add_emoticon)

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
pattern_emoticon = pattern_emoticon.replace('\{','\\{')
pattern_emoticon = pattern_emoticon.replace('\}','\\}')
pattern_emoticon = pattern_emoticon.replace('<','\\>')
pattern_emoticon = pattern_emoticon.replace('>','\\>')
pattern_emoticon = pattern_emoticon.replace('?','\\?')

def _get_emoticon_counts(x):
    emoticon_pattern = re.compile(u'(' + pattern_emoticon + u')')
    emoticon = [i for i in emoticon_pattern.findall(x) if i]
    return len(emoticon)


def _extract_emoticons(x):
    emoticon_pattern = re.compile(u'(' + pattern_emoticon + u')')
    emoticon = [EMOTICONS_EMO[i] for i in emoticon_pattern.findall(x) if i]
    return emoticon


# Convert emoticons into word
def _convert_emoticons(x):
    for emot in EMOTICONS_EMO:
        x = x.replace(emot, "_".join(EMOTICONS_EMO[emot].replace(",","").replace(":","").split()))
    return x


# Remove emoticons
def _remove_emoticons(x):
    for emot in EMOTICONS_EMO:
        x = x.replace(emot, " ")
    return x


# Count emoji
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

def _extract_emojis(x):
    emoji_pattern = re.compile(u'(' + pattern_emoji + u')')
    emoji = [UNICODE_EMOJI[i] for i in emoji_pattern.findall(x) if i]
    return emoji

# Convert emoji into word
def _convert_emojis(x):
    for emot in UNICODE_EMOJI:
        x = x.replace(emot, "_".join(UNICODE_EMOJI[emot].replace(",","").replace(":","").split()))
    return x


# Remove emoji 
def _remove_emojis(x):
    for emot in UNICODE_EMOJI:
        x = x.replace(emot, " ")
    return x


# Lemmatize using spacy
lemmatizer = nlp.get_pipe("lemmatizer")
def _get_lemmatize_words(x):
    return " ".join([token.lemma_ for token in nlp(x)])


# Get NER using spacy
def _get_ner(x):
    return " ".join([ent.label_ for ent in nlp(x).ents])


# Count NER using spacy
def _get_ner_counts(x,pos_tag):
    return len([t for t in x.split() if t == pos_tag])


# Get POS tag using spacy
def _get_pos_tag(x):
    return " ".join([token.pos_ for token in nlp(x)])


# Count POS tag using spacy
def _get_pos_tag_counts(x,pos_tag):
    return len([t for t in x.split() if t == pos_tag])


# Resolve all Internet Slangs
with open('SLANG_TOP.pkl', 'rb') as fp:
    SLANG_TOP = pickle.load(fp)
    
def _slang_resolution(x):
    clean_text = []
    for text in x.split():
        if text in list(SLANG_TOP.keys()):
            for key in SLANG_TOP:
                value = SLANG_TOP[key]
                if text == key:
                    clean_text.append(text.replace(key,value))
                else:
                    continue
        else:
            clean_text.append(text)
    return " ".join(clean_text)


# Remove space between single characters
def _remove_space_single_chars(x):
    
    '''
    ----------------
     Decipher regex
    ----------------
    (?i)          # set flags for this block (case-insensitive)
    (?<=          # look behind to see if there is:
      \b          #   the boundary between a word char (\w) and not a word char
      [a-z]       #   any character of: 'a' to 'z'
    )             # end of look-behind
                  # ' '
    (?=           # look ahead to see if there is:
      [a-z]       #   any character of: 'a' to 'z'
      \b          #   the boundary between a word char (\w) and not a word char
    )             # end of look-ahead
    
    '''
    temp = re.sub(r'(?i)(?<=\b[a-z]) (?=[a-z]\b)', '', x) 
    return temp