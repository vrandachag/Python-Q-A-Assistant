"""
Preprocesses the Kaggle Stack Overflow Python dataset into a clean JSON file.

Dataset: https://www.kaggle.com/datasets/stackoverflow/pythonquestions
Expected files in ./data/raw/:
  - Questions.csv
  - Answers.csv

Run: python preprocess.py
"""

import pandas as pd
import json
import re
import os
from html import unescape

RAW_DIR = r"C:\Users\chagv\Downloads\archive\python-qa-assistant\python-qa-assistant\data\raw"
OUTPUT_PATH = r"C:\Users\chagv\Downloads\archive\python-qa-assistant\python-qa-assistant\data/processed_qa.json"
MAX_RECORDS = 10000  # Adjust based on your compute/budget

def clean_text(text: str) -> str:
    """Remove HTML tags and normalize whitespace."""
    text = unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess():
    print("Loading questions...")
    questions = pd.read_csv(
        f"{RAW_DIR}/Questions.csv",
        encoding="latin-1",
        usecols=["Id", "Title", "Body", "Score"],
        nrows=MAX_RECORDS * 2
    )

    print("Loading answers...")
    answers = pd.read_csv(
        f"{RAW_DIR}/Answers.csv",
        encoding="latin-1",
        usecols=["ParentId", "Body", "Score"]
    )

    # Keep only accepted/top answers per question
    top_answers = (
        answers.sort_values("Score", ascending=False)
        .groupby("ParentId")
        .first()
        .reset_index()
        .rename(columns={"ParentId": "Id", "Body": "AnswerBody"})
    )

    merged = questions.merge(top_answers[["Id", "AnswerBody"]], on="Id", how="inner")

    # Filter: only questions with meaningful score and an answer
    merged = merged[merged["Score"] >= 1].dropna(subset=["AnswerBody"])
    merged = merged.head(MAX_RECORDS)

    records = []
    for _, row in merged.iterrows():
        records.append({
            "title": clean_text(row["Title"]),
            "body": clean_text(row["Body"]),
            "answer": clean_text(row["AnswerBody"]),
            "score": int(row["Score"])
        })

    os.makedirs("./data", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Saved {len(records)} records to {OUTPUT_PATH}")

if __name__ == "__main__":
    preprocess()
