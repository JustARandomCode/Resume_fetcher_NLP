from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.exceptions import NotFittedError


class TFIDFRetrieval:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.resume_vectors = None
        self.resume_texts = None

    def fit(self, resume_texts):
        self.resume_texts = resume_texts
        self.resume_vectors = self.vectorizer.fit_transform(resume_texts)

    def search(self, query, top_k=5):
        if self.resume_vectors is None:
            raise NotFittedError("TFIDFRetrieval: call fit() before search().")

        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.resume_vectors).flatten()

        ranked_indices = similarities.argsort()[::-1][:top_k]

        results = []
        for idx in ranked_indices:
            results.append({
                "index": int(idx),
                "score": float(similarities[idx])
            })

        return results