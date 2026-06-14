# 🐍 Python Q&A Assistant

An AI-powered Python programming Q&A system built with RAG (Retrieval-Augmented Generation), backed by Stack Overflow data.

> Built for the Analytics Vidhya AI Engineer Assessment

**Live Demo:** Local deployment only

---

## Architecture

```
User Question
     │
     ▼
FastAPI /ask endpoint
     │
     ▼
LangChain RAG Chain
  ├── Retriever: ChromaDB Vector Store
  │   └── Embeddings: Sentence Transformers (all-MiniLM-L6-v2)
  └── Response: Top-k Retrieved StackOverflow Answers
     │
     ▼
Answer + Source Documents
```

---

## Setup

### 1. Clone & install
```bash
git clone https://github.com/your-username/python-qa-assistant
cd python-qa-assistant
pip install -r requirements.txt
```

### 2. Set environment variables
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Download & preprocess dataset
- Download from [Kaggle](https://www.kaggle.com/datasets/stackoverflow/pythonquestions)
- Place `Questions.csv` and `Answers.csv` in `./data/raw/`
- Run:
```bash
python preprocess.py
```

### 4. Build the vector index (first run only)
The index is built automatically on first startup. Subsequent startups load from `./chroma_db/`.

### 5. Run the API
```bash
uvicorn app.main:app --reload --port 8000
```

API docs available at: `http://localhost:8000/docs`

---

## API Endpoints

### `POST /ask`
```json
// Request
{ "question": "How do I reverse a list in Python?" }

// Response
{
  "question": "How do I reverse a list in Python?",
  "answer": "You can reverse a list using...",
  "sources": [
    { "title": "...", "score": 42.0, "snippet": "..." }
  ],
  "latency_ms": 823.5
}
```

### `GET /health`
```json
{ "status": "ok", "pipeline_ready": true }
```

---

## Running Tests
```bash
pytest tests/test_api.py -v
```

---

## Deployment (Hugging Face Spaces)

This project was developed and tested locally using FastAPI and ChromaDB.

Run the application:

python -m uvicorn app.main:app --reload

API documentation:

http://localhost:8000/docs
---

## Project Structure
```
python-qa-assistant/
├── app/
│   ├── main.py          # FastAPI app
│   ├── models.py        # Pydantic schemas
│   └── rag_pipeline.py  # RAG logic
├── tests/
│   └── test_api.py      # Pytest test suite
├── notebooks/
│   └── test_results.ipynb  # 8 test queries documented
├── data/
│   └── raw/             # Place Kaggle CSVs here
├── preprocess.py        # Dataset preprocessing
├── requirements.txt
├── Dockerfile
└── .env.example
```
