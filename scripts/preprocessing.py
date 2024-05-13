import string
import contractions

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


nltk.download('stopwords')
nltk.download('wordnet')


def base_text_preprocessing(text_col):
    text_col = text_col.str.lower()
    text_col = text_col.str.replace(r'[0-9]+', '', regex=True)

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text_col = text_col.apply(lambda text_i: ' '.join([
        contractions.fix(lemmatizer.lemmatize(word))
        for word in text_i.split() if word not in stop_words]))
    # contractions: e.g., "don't" to "do not"

    text_col = text_col.str.translate(str.maketrans('', '', string.punctuation))
    return text_col
