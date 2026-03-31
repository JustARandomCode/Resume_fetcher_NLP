import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def _ensure_vader():
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)


_ensure_vader()
_sia = SentimentIntensityAnalyzer()


def analyze_sentiment(text: str) -> dict:
    """
    Run VADER sentiment analysis on the given text.

    Returns:
        {
            "compound": float,   # -1.0 (most negative) to +1.0 (most positive)
            "label":    str,     # "Positive" | "Neutral" | "Negative"
            "pos":      float,   # proportion of positive tokens
            "neu":      float,   # proportion of neutral tokens
            "neg":      float    # proportion of negative tokens
        }
    """
    if not text or not text.strip():
        return {"compound": 0.0, "label": "Neutral", "pos": 0.0, "neu": 1.0, "neg": 0.0}

    scores = _sia.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return {
        "compound": round(compound, 4),
        "label": label,
        "pos": round(scores["pos"], 4),
        "neu": round(scores["neu"], 4),
        "neg": round(scores["neg"], 4),
    }