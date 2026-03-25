from sentence_transformers import SentenceTransformer, util


class EmbeddingRetrieval:
    def __init__(self, model):
        # Accept a pre-loaded model so app.py can cache it via st.cache_resource
        # and avoid reloading it on every Streamlit rerun.
        self.model = model
        self.resume_embeddings = None

    def fit(self, resume_texts):
        self.resume_embeddings = self.model.encode(
            resume_texts,
            convert_to_tensor=True,
            show_progress_bar=False
        )

    def search(self, query, top_k=5):
        if self.resume_embeddings is None:
            raise RuntimeError("EmbeddingRetrieval: call fit() before search().")

        # query must be raw natural language — do NOT preprocess before passing here
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        similarities = util.cos_sim(query_embedding, self.resume_embeddings)[0]

        top_results = similarities.argsort(descending=True)[:top_k]

        results = []
        for idx in top_results:
            results.append({
                "index": int(idx),
                "score": float(similarities[idx])
            })

        return results