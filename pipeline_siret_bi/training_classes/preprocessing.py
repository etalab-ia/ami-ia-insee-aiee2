import re
import string
import unidecode
from nltk.stem.snowball import FrenchStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def trim(text):
    """
    Strip text from start and end spaces

    :param text: str
    :returns: str
    """
    return text.strip()

def text_lowercase(text):
    """
    set text to lowercase

    :param text: str
    :returns: str
    """
    return text.lower()

def keep_alphanumeric(text):
    """
    Remove non alphanumeric chars

    :param text: str
    :returns: str
    """
    pattern = re.compile('[\W_]+', re.UNICODE)
    return pattern.sub(' ', text)

def letters_only(text):
    """
    Remove all non letters chars

    :param text: str
    :returns: str
    """
    return re.sub('[^a-zA-Z]+', ' ', text)

def remove_punctuation(text):
    """
    remove punctuation from text

    :param text: str
    :returns: str
    """
    punctuation = string.punctuation + '’—°€'
    for punct in punctuation:
        text = text.replace(punct, ' ')
    return text

def remove_accents(text):
    """
    replace accented chars by non-accented equivalent

    :param text: str
    :returns: str
    """
    return unidecode.unidecode(text)

def remove_digits(text):
    """
    Remoce all digits

    :param text: str
    :returns: str
    """
    for digit in string.digits:
        text = text.replace(digit, ' ')
    return text

def tokenize(text):
    """
    tokenize using nltk.tokenize.word_tokenize

    :param text: str
    :returns: str
    """
    text = word_tokenize(text)
    return text

def clean(text):
    """
    Remove punctuation and accents

    :param text: str
    :returns: str
    """
    text = remove_punctuation(text)
    text = remove_accents(text)
    text = text_lowercase(text)
    return text

def remove_stopwords(text):
    """
    remove french stopwords (nltf.corpus.stopwords)

    :param text: str
    :returns: str
    """
    stop_words = set(stopwords.words('french'))
    words = text.split(" ")
    text = [i for i in words if not i in stop_words]
    text = " ".joint(text)
    return text

def preprocessing(text):
    """
    set to lowercase, remove punctuation, accents, digits
    tokenize, remove french stopwords, and finally stems with a FrenchStemmer

    :param text: str
    :returns: str
    """
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
    """
    set to lowercase, remove punctuation, and all non letters
    tokenize, remove french stopwords, and finally stems with a FrenchStemmer
    
    :param text: str
    :returns: str
    """
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
    """
    set to lowercase, remove punctuation, accents, digits
    tokenize, remove french stopwords
    
    :param text: str
    :returns: str
    """
    text = text_lowercase(text)
    text = remove_accents(text)
    text = letters_only(text)
    text = tokenize(text)
    text = remove_stopwords(text)
    text = ' '.join(text)
    return text