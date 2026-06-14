from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.models import QuestionRequest, AnswerResponse
from app.rag_pipeline import RAGPipeline
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Python Q&A Assistant",
    description="AI-powered Python Q&A using Stack Overflow data + RAG",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rag: RAGPipeline = None

@app.on_event("startup")
async def startup_event():
    global rag
    logger.info("Initializing RAG pipeline...")
    rag = RAGPipeline()
    rag.load_or_build_index()
    logger.info("RAG pipeline ready.")

@app.get("/health")
async def health():
    return {"status": "ok", "pipeline_ready": rag is not None}

@app.post("/ask", response_model=AnswerResponse)
async def ask(request: QuestionRequest):
    if not rag:
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    start = time.time()
    try:
        result = rag.answer(request.question)
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate answer")

    return AnswerResponse(
        question=request.question,
        answer=result["answer"],
        sources=result["sources"],
        latency_ms=round((time.time() - start) * 1000, 2)
    )
