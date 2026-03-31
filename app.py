import os
import streamlit as st
from src.utils import build_resume_dataframe
from src.preprocess import preprocess_text
from src.tfidf_retrieval import TFIDFRetrieval
from src.embedding_retrieval import EmbeddingRetrieval
from src.sentiment_analysis import analyze_sentiment
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Resume Retrieval System", layout="wide")

RESUME_FOLDER = "data/resumes"

SENTIMENT_COLORS = {
    "Positive": "#28a745",
    "Neutral":  "#6c757d",
    "Negative": "#dc3545",
}


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

# ── Sidebar: Sentiment Filter ────────────────────────────────────────────────
st.sidebar.header("Filters")

sentiment_options = ["All", "Positive", "Neutral", "Negative"]
selected_sentiment = st.sidebar.selectbox("Filter by Resume Sentiment", sentiment_options)

if selected_sentiment != "All":
    filtered_df = df[df["sentiment_label"] == selected_sentiment].reset_index(drop=True)
    if filtered_df.empty:
        st.sidebar.warning(f"No resumes with '{selected_sentiment}' sentiment.")
        filtered_df = df  # fallback to full set
    else:
        st.sidebar.info(f"{len(filtered_df)} resume(s) match.")
else:
    filtered_df = df

# ── Sidebar: Sentiment Overview ──────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("Sentiment Overview")
sentiment_counts = df["sentiment_label"].value_counts()
for label, count in sentiment_counts.items():
    color = SENTIMENT_COLORS.get(str(label), "#333")
    st.sidebar.markdown(
        f"<span style='color:{color}; font-weight:bold;'>{label}</span>: {count}",
        unsafe_allow_html=True
    )

# ── Main: Query Sentiment Analysis ───────────────────────────────────────────
st.markdown("---")
st.subheader("Query Sentiment")
st.caption("VADER analyses the tone of your search prompt in real time.")

# ── Main: Retrieval Controls ─────────────────────────────────────────────────
method = st.selectbox("Select Retrieval Method", ["TF-IDF", "Embeddings"])
query  = st.text_input("Enter your prompt", placeholder="e.g. Python developer with NLP and Flask experience")

# Show query sentiment live as user types
if query.strip():
    q_sentiment = analyze_sentiment(query)
    q_color = SENTIMENT_COLORS.get(q_sentiment["label"], "#333")
    st.markdown(
        f"**Query Sentiment:** "
        f"<span style='color:{q_color}; font-weight:bold;'>{q_sentiment['label']}</span> "
        f"(compound score: `{q_sentiment['compound']}`)",
        unsafe_allow_html=True,
    )

if len(filtered_df) == 0:
    st.warning("No resumes match the current filter.")
    st.stop()
else:
    top_k = st.slider("Top K resumes", 1, min(10, len(filtered_df)), min(5, len(filtered_df)))

if st.button("Search") and query:

    if method == "TF-IDF":
        processed_query = preprocess_text(query)
        retriever = get_tfidf_retriever(filtered_df)
        results = retriever.search(processed_query, top_k=top_k)
    else:
        model = load_embedding_model()
        retriever = get_embedding_retriever(filtered_df, model)
        results = retriever.search(query, top_k=top_k)

    st.subheader("Top Matching Resumes")

    for rank, result in enumerate(results, start=1):
        row = filtered_df.iloc[result["index"]]

        sentiment_label    = row["sentiment_label"]
        sentiment_compound = row["sentiment_compound"]
        sentiment_pos      = row["sentiment_pos"]
        sentiment_neu      = row["sentiment_neu"]
        sentiment_neg      = row["sentiment_neg"]
        badge_color        = SENTIMENT_COLORS.get(sentiment_label, "#333")

        st.markdown(f"### Rank {rank}")
        st.write(f"**File Name:** {row['file_name']}")
        st.write(f"**Similarity Score:** {result['score']:.4f}")

        # Sentiment badge + breakdown
        st.markdown(
            f"**Resume Sentiment:** "
            f"<span style='background-color:{badge_color}; color:white; padding:2px 8px; "
            f"border-radius:4px; font-weight:bold;'>{sentiment_label}</span> "
            f"&nbsp; Compound: `{sentiment_compound}` &nbsp;|&nbsp; "
            f"Pos: `{sentiment_pos}` &nbsp;|&nbsp; "
            f"Neu: `{sentiment_neu}` &nbsp;|&nbsp; "
            f"Neg: `{sentiment_neg}`",
            unsafe_allow_html=True,
        )

        st.write(f"**Preview:** {row['raw_text'][:1000]}...")
        st.markdown("---")