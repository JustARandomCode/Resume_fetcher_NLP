import os
import streamlit as st
from src.utils import build_resume_dataframe
from src.preprocess import preprocess_text
from src.tfidf_retrieval import TFIDFRetrieval
from src.embedding_retrieval import EmbeddingRetrieval
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Resume Retrieval System", layout="wide")

RESUME_FOLDER = "data/resumes"


@st.cache_data
def load_resume_dataframe(folder):
    return build_resume_dataframe(folder)


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def get_tfidf_retriever(_df):
    retriever = TFIDFRetrieval()
    retriever.fit(_df["clean_text"].tolist())
    return retriever


@st.cache_resource
def get_embedding_retriever(_df, _model):
    retriever = EmbeddingRetrieval(model=_model)
    retriever.fit(_df["clean_text"].tolist())
    return retriever


# ── UI ──────────────────────────────────────────────────────────────────────

st.title("Resume Retrieval System using NLP")
st.write("Retrieve the most relevant resumes based on a user prompt.")

# Ensure resume folder exists
if not os.path.exists(RESUME_FOLDER):
    st.error(
        f"Resume folder `{RESUME_FOLDER}` not found. "
        "Create the folder and add `.pdf`, `.docx`, or `.txt` resume files, then refresh."
    )
    st.stop()

df = load_resume_dataframe(RESUME_FOLDER)

if df.empty:
    st.warning(
        f"No resumes were loaded from `{RESUME_FOLDER}`. "
        "Add supported files (.pdf, .docx, .txt) and refresh."
    )
    st.stop()

st.success(f"{len(df)} resume(s) loaded.")

method = st.selectbox("Select Retrieval Method", ["TF-IDF", "Embeddings"])
query = st.text_input("Enter your prompt", placeholder="e.g. Python developer with NLP and Flask experience")
if len(df) == 0:
    st.warning("No resumes found in 'data/resumes'. Please add resume files.")
    st.stop()
else:
    top_k = st.slider("Top K resumes", 1, min(10, len(df)), min(5, len(df)))

if st.button("Search") and query:

    if method == "TF-IDF":
        # TF-IDF works on bag-of-words: use preprocessed query
        processed_query = preprocess_text(query)
        retriever = get_tfidf_retriever(df)
        results = retriever.search(processed_query, top_k=top_k)

    else:
        # Sentence transformers require natural language: use raw query
        model = load_embedding_model()
        retriever = get_embedding_retriever(df, model)
        results = retriever.search(query, top_k=top_k)

    st.subheader("Top Matching Resumes")

    for rank, result in enumerate(results, start=1):
        row = df.iloc[result["index"]]
        st.markdown(f"### Rank {rank}")
        st.write(f"**File Name:** {row['file_name']}")
        st.write(f"**Similarity Score:** {result['score']:.4f}")
        st.write(f"**Preview:** {row['raw_text'][:1000]}...")
        st.markdown("---")