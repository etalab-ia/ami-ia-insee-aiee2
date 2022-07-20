import os
import re
import string
import unidecode
import nltk
from nltk.stem.snowball import FrenchStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

if os.environ.get('http_proxy', None):
    nltk.set_proxy(os.environ['http_proxy'])
nltk.download('stopwords')
nltk.download('punkt')

def trim(text):
    return text.strip()

def text_lowercase(text):
    return text.lower()

def keep_alphanumeric(text):
    pattern = re.compile('[\W_]+', re.UNICODE)
    return pattern.sub(' ', text)

def letters_only(text):
    return re.sub('[^a-zA-Z]+', ' ', text)

def remove_punctuation(text):
    punctuation = string.punctuation + '’—°€'
    for punct in punctuation:
        text = text.replace(punct, ' ')
    return text

def remove_accents(text):
    return unidecode.unidecode(text)

def remove_digits(text):
    for digit in string.digits:
        text = text.replace(digit, ' ')
    return text

def tokenize(text):
    text = word_tokenize(text)
    return text

def clean(text):
    text = remove_punctuation(text)
    text = remove_accents(text)
    text = text_lowercase(text)
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('french'))   
    if isinstance(text, str):
        words = text.split(" ")
    else:
        words = text
    text = [i for i in words if not i in stop_words]
    if isinstance(text, str):
        return " ".join(text)
    return text

def preprocessing(text):
    text = text_lowercase(text)
    text = remove_punctuation(text)
    text = remove_accents(text)
    text = remove_digits(text)
    text = tokenize(text)
    text = remove_stopwords(text)
    text = ' '.join(text)
    stemmer = FrenchStemmer()
    text = stemmer.stem(text)
    return text

def preprocessing_all(text):
    text = text_lowercase(text)
    text = remove_accents(text)
    text = letters_only(text)
    text = tokenize(text)
    text = remove_stopwords(text)
    text = ' '.join(text)
    stemmer = FrenchStemmer()
    text = stemmer.stem(text)
    return text

def preprocessing_no_stemmer(text):
    text = text_lowercase(text)
    text = remove_accents(text)
    text = letters_only(text)
    text = tokenize(text)
    text = remove_stopwords(text)
    text = ' '.join(text)
    return text