import os
import pandas as pd
from src.extract_text import extract_resume_text
from src.preprocess import preprocess_text

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def build_resume_dataframe(resume_folder):
    if not os.path.exists(resume_folder):
        print(f"Warning: Resume folder '{resume_folder}' does not exist.")
        return pd.DataFrame(columns=["file_name", "raw_text", "clean_text"])

    data = []

    for file_name in os.listdir(resume_folder):
        ext = os.path.splitext(file_name)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue

        file_path = os.path.join(resume_folder, file_name)

        if not os.path.isfile(file_path):
            continue

        raw_text = extract_resume_text(file_path)

        if not raw_text.strip():
            print(f"Warning: No text extracted from '{file_name}', skipping.")
            continue

        clean_text = preprocess_text(raw_text)

        data.append({
            "file_name": file_name,
            "raw_text": raw_text,
            "clean_text": clean_text
        })

    return pd.DataFrame(data)