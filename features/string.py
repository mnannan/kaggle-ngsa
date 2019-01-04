from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stem = PorterStemmer()
english_stopwords = set(stopwords.words('english'))


def clean_string(string: str) -> str:
    """
    Remove stopwords and stem the string
    """
    string = stem.stem(string)
    words = []
    for word in string.split():
        if word not in english_stopwords:
            words.append(word)
    return ' '.join(words)


def overlap(string1: str, string2: str) -> int:
    """ Given two strings returns the number of common words"""
    return len(set(string1).intersection(set(string2)))
