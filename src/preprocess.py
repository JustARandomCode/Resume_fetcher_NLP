import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def _download_nltk_resource(resource, name):
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(name, quiet=True)


_download_nltk_resource("corpora/stopwords", "stopwords")
_download_nltk_resource("corpora/wordnet", "wordnet")
_download_nltk_resource("corpora/omw-1.4", "omw-1.4")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    tokens = text.split()

    cleaned_tokens = []
    for token in tokens:
        if token not in stop_words and len(token) > 2:
            cleaned_tokens.append(lemmatizer.lemmatize(token))

    return " ".join(cleaned_tokens)