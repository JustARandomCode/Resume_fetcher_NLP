# NLP Resume Ranker

A Streamlit-based web application that uses NLP techniques to rank resumes against a job description. Upload resumes, enter a job description, and get the top matching candidates ranked by relevance.

---

## Features

- Upload and parse resumes in PDF or DOCX format
- NLP-powered similarity scoring using Sentence Transformers
- TF-IDF based keyword extraction and matching
- Interactive Streamlit UI with adjustable Top-K slider
- Supports batch processing of multiple resumes

---

## Tech Stack

| Library | Purpose |
|---|---|
| `streamlit` | Web UI |
| `pandas` / `numpy` | Data handling |
| `scikit-learn` | TF-IDF vectorization |
| `nltk` | Text preprocessing |
| `pdfplumber` | PDF text extraction |
| `python-docx` | DOCX text extraction |
| `sentence-transformers` | Semantic similarity scoring |
| `torch` | Backend for sentence transformers |

---

## Project Structure

```
NLP_project/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── data/
│   └── resumes/            # Place your resume files here (PDF/DOCX)
└── src/
    ├── __init__.py
    ├── extract_text.py     # Resume text extraction logic
    ├── preprocess.py       # Text cleaning and preprocessing
    ├── tfidf_re(...).py    # TF-IDF ranking logic
    └── utils.py            # Helper functions (build_resume_dataframe, etc.)
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/NLP_project.git
cd NLP_project
```

### 2. (Recommended) Create a virtual environment

```bash
# Using Anaconda
conda create -n nlp_env python=3.10
conda activate nlp_env
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK data (first-time setup)

```python
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

---

## Running the App

> **Important:** Always run Streamlit apps using `streamlit run`, not the VS Code Play button.

```bash
streamlit run app.py
```

Then open your browser at: [http://localhost:8501](http://localhost:8501)

---

## Adding Resumes

Place your resume files inside the `data/resumes/` folder before launching the app:

```bash
mkdir -p data/resumes
# Then copy your PDF or DOCX resumes into data/resumes/
```

---

## Common Issues

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: streamlit` | Wrong Python interpreter in VS Code | Select the Anaconda interpreter via `Ctrl+Shift+P` → *Python: Select Interpreter* |
| `Slider min_value must be less than max_value` | No resumes found in `data/resumes/` | Add resume files to the folder |
| `Warning: Resume folder does not exist` | `data/resumes/` directory missing | Run `mkdir data\resumes` |

---

## Requirements

- Python 3.9+
- Anaconda (recommended) or any Python environment
- VS Code with the Python extension (optional)

---

## License

This project is for educational and personal use.
