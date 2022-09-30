from preprocess_text import utils

__version__ = '1.7.2'

def get_wordcounts(x):
    return utils._get_wordcounts(x)

def get_char_counts(x):
    return utils._get_char_counts(x)

def get_avg_wordlength(x):
    return utils._get_avg_wordlength(x)

def get_stopwords_counts(x):
    return utils._get_stopwords_counts(x)

def get_punc_counts(x):
    return utils._get_punc_counts(x)

def get_hashtag_counts(x):
    return utils._get_hashtag_counts(x)

def get_mention_counts(x):
    return utils._get_mention_counts(x)

def get_digit_counts(x):
    return utils._get_digit_counts(x)

def get_uppercase_counts(x):
    return utils._get_uppercase_counts(x)

def cont_exp(x):
    return utils._cont_exp(x)

def get_emails(x):
    return utils._get_emails(x)

def remove_emails(x):
    return utils._remove_emails(x)

def get_urls(x):
    return utils._get_urls(x)

def remove_urls(x):
    return utils._remove_urls(x)

def remove_mention(x):
    return utils._remove_mention(x)

def remove_special_chars(x):
    return utils._remove_special_chars(x)

def remove_elongated_chars(x):
    return utils._remove_elongated_chars(x)

def remove_html_tags(x):
    return utils._remove_html_tags(x)

def remove_accented_chars(x):
    return utils._remove_accented_chars(x)

def remove_numeric(x):
    return utils._remove_numeric(x)

def remove_stopwords(x,keep_pronoun):
    return utils._remove_stopwords(x,keep_pronoun)

def make_base(x):
    return utils._make_base(x)

def get_value_counts(df,col):
    return utils._get_value_counts(df,col)

def remove_common_words(x, freq, n=20):
    return utils._remove_common_words(x, freq, n)

def remove_rarewords(x, freq, n=20):
    return utils._remove_rarewords(x, freq, n)

def spelling_correction(x):
    return utils._spelling_correction(x)

def get_emoticon_counts(x):
    return utils._get_emoticon_counts(x)

def convert_emoticons(x):
    return utils._convert_emoticons(x)

def extract_emoticons(x):
    return utils._extract_emoticons(x)

def remove_emoticons(x):
    return utils._remove_emoticons(x)

def get_emoji_counts(x):
    return utils._get_emoji_counts(x)

def convert_emojis(x):
    return utils._convert_emojis(x)

def extract_emojis(x):
    return utils._extract_emojis(x)

def remove_emojis(x):
    return utils._remove_emojis(x)

def get_lemmatize_words(x):
    return utils._get_lemmatize_words(x)

def get_ner(x):
    return utils._get_ner(x)

def get_ner_counts(x,pos_tag):
    return utils._get_ner_counts(x,pos_tag)

def get_pos_tag(x):
    return utils._get_pos_tag(x)

def get_pos_tag_counts(x,pos_tag):
    return utils._get_pos_tag_counts(x,pos_tag)

def slang_resolution(x):
    return utils._slang_resolution(x)

def remove_space_single_chars(x):
    return utils._remove_space_single_chars(x)